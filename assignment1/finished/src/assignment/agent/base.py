"""The domain-independent ReAct loop shared by both agents.

Part 1 completes the generic loop here; the two subclasses in this package
supply only their own tools and tool executors.
"""

from __future__ import annotations

from copy import deepcopy
import json
import logging
import math
import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from openai import OpenAI

from assignment.env import Environment
from assignment.agent.tools import INVOKE_SKILL_TOOL

load_dotenv()
logger = logging.getLogger(__name__)

DEFAULT_COMPACTION_KEEP_RECENT_STEPS = 1
DEFAULT_COMPACTION_MAX_TOKENS = 1_200
MAX_OBSERVATION_CHARS = 10_000

COMPACTION_SYSTEM_PROMPT = """You maintain the working memory of a software agent.

Rewrite the transcript you are given as concise factual notes the agent can act
on once the raw messages are gone. Preserve, where the transcript establishes
them:

- the objective, and any constraints or requirements on it
- files inspected or edited, and what each edit changed
- commands run and the concrete result of each
- approaches that failed, and why they failed
- tests run and their outcomes
- open blockers and unanswered questions
- the next action the agent was about to take

Record what a command established, not what it printed. Keep identifiers exact:
paths, symbols, line numbers, error messages, and test names. State nothing the
transcript does not support, and omit whatever it never established. Answer with
the notes alone, as short bullets, with no preamble."""


class StepLimitError(Exception):
    """Raised when an agent exhausts its model-call budget."""


def format_tool_output(output: dict[str, Any]) -> str:
    """Format a terminal result as a compact, tagged model observation."""

    elements: list[str] = []
    for key in sorted(output):
        value = output[key]
        if isinstance(value, str) and len(value) > MAX_OBSERVATION_CHARS:
            # Leave room for the elision notice so the formatted value itself,
            # not just its retained source slices, stays below the limit.
            retained_at_each_end = 4_900
            omitted = len(value) - (2 * retained_at_each_end)
            value = (
                f"{value[:retained_at_each_end]}\n"
                f"[{omitted} characters elided; read a narrower range]\n"
                f"{value[-retained_at_each_end:]}"
            )
        elements.append(f"<{key}>{value}</{key}>")
    return "\n".join(elements)


def rough_message_tokens(messages: list[dict[str, Any]]) -> int:
    """Estimate prompt tokens without a provider-specific tokenizer."""

    serialized = json.dumps(messages, ensure_ascii=False, separators=(",", ":"))
    return max(1, math.ceil(len(serialized) / 4))


class Agent:
    """Base class for a ReAct agent with pluggable tools."""

    def __init__(
        self,
        environment: Environment,
        model: str | None = None,
        logs_save_path: str | None = None,
        step_limit: int = 100,
        skills_path: str | None = None,
        auto_stop_environment: bool = True,
        compact_threshold_tokens: int | None = None,
        compaction_keep_recent_steps: int = DEFAULT_COMPACTION_KEEP_RECENT_STEPS,
        compaction_max_tokens: int = DEFAULT_COMPACTION_MAX_TOKENS,
    ):
        self.env = environment
        self.model = model or os.environ.get("OPENAI_MODEL")
        if not self.model:
            raise RuntimeError("OPENAI_MODEL is not set.")

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        base_url = os.environ.get("OPENAI_BASE_URL")
        if not base_url:
            raise RuntimeError("OPENAI_BASE_URL is not set.")
        try:
            max_retries = int(os.environ.get("OPENAI_MAX_RETRIES", "5"))
        except ValueError as exc:
            raise RuntimeError("OPENAI_MAX_RETRIES must be an integer.") from exc
        if max_retries < 0:
            raise RuntimeError("OPENAI_MAX_RETRIES must be non-negative.")

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            max_retries=max_retries,
        )

        self.logs_save_path = logs_save_path
        self.step_limit = step_limit
        self.auto_stop_environment = auto_stop_environment
        if compact_threshold_tokens is not None and compact_threshold_tokens <= 0:
            raise ValueError("compact_threshold_tokens must be positive or None")
        if (
            compaction_keep_recent_steps is not None
            and compaction_keep_recent_steps < 1
        ):
            raise ValueError("compaction_keep_recent_steps must be at least 1")
        if compaction_max_tokens is not None and compaction_max_tokens < 1:
            raise ValueError("compaction_max_tokens must be positive")
        # A None threshold turns compaction off. The other two settings then
        # describe a compaction that never happens, so fall back to the
        # defaults rather than leaving a None for later code to trip over.
        self.compact_threshold_tokens = compact_threshold_tokens
        self.compaction_keep_recent_steps = (
            DEFAULT_COMPACTION_KEEP_RECENT_STEPS
            if compaction_keep_recent_steps is None
            else compaction_keep_recent_steps
        )
        self.compaction_max_tokens = (
            DEFAULT_COMPACTION_MAX_TOKENS
            if compaction_max_tokens is None
            else compaction_max_tokens
        )

        # Each agent supplies its own opening messages: the standing
        # instructions, and the task statement that starts the run.
        self.system_prompt: str = ""
        self.task_prompt: str = ""

        self.api_prompts: list[list[dict[str, Any]]] = []
        self.api_responses: list[dict[str, Any]] = []
        self.compaction_events: list[dict[str, Any]] = []
        self.tools: list[dict[str, Any]] = []
        self.finished = False
        self.steps_taken = 0

        self.skills_path = Path(skills_path) if skills_path is not None else None
        self.skills: dict[str, dict[str, str]] = (
            self.load_skills(self.skills_path) if self.skills_path is not None else {}
        )

        if self.skills:
            self.tools.append(INVOKE_SKILL_TOOL)

        # The run so far, in the order the model should see it: each assistant
        # message as returned by `process_response`, followed by the `tool`
        # messages answering the calls it made. The loop appends to this; the
        # opening system and task messages are not stored here so that they
        # stay verbatim no matter what happens to the history.
        self.history: list[dict[str, Any]] = []

        # Compaction replaces the oldest `compacted_history_length` entries of
        # the history with `context_summary`, the working memory standing in
        # for them.
        self.context_summary: str = ""
        self.compacted_history_length: int = 0

    def load_skills(self, skills_path: Path) -> dict[str, dict[str, str]]:
        """Load the skill folders exposed to this agent."""

        if not skills_path.is_dir():
            raise ValueError(f"No skills directory at {skills_path}.")

        skills: dict[str, dict[str, str]] = {}
        for directory in sorted(item for item in skills_path.iterdir() if item.is_dir()):
            skill_file = directory / "SKILL.md"
            if not skill_file.is_file():
                raise ValueError(f"Skill folder {directory} has no SKILL.md.")

            content = skill_file.read_text()
            lines = content.splitlines()
            if not lines or lines[0].strip() != "---":
                raise ValueError(
                    f"{skill_file} does not open with a `---` frontmatter block."
                )
            for end, line in enumerate(lines[1:], start=1):
                if line.strip() == "---":
                    break
            else:
                raise ValueError(f"{skill_file} never closes its frontmatter with `---`.")

            metadata = "\n".join(lines[1:end]).strip()
            try:
                frontmatter = yaml.safe_load(metadata)
            except yaml.YAMLError as exc:
                raise ValueError(f"{skill_file} has unparseable frontmatter: {exc}") from exc
            if not isinstance(frontmatter, dict):
                raise ValueError(f"{skill_file} frontmatter is not a mapping of keys.")

            # The catalog in the prompt is built from these two fields, so a
            # skill missing either one could never be advertised or invoked.
            name = frontmatter.get("name")
            description = frontmatter.get("description")
            if not isinstance(name, str) or not name.strip():
                raise ValueError(f"{skill_file} frontmatter has no `name`.")
            if not isinstance(description, str) or not description.strip():
                raise ValueError(f"{skill_file} frontmatter has no `description`.")

            name = name.strip()
            if name in skills:
                raise ValueError(
                    f"Two skill folders are both named {name!r}; names must be unique."
                )
            skills[name] = {"metadata": metadata, "content": content}

        return skills

    def query_language_model(self) -> dict[str, Any]:
        """Send one tool-enabled Chat Completions request and normalize it."""

        messages = self.build_prompt()
        self.api_prompts.append(deepcopy(messages))
        step_number = self.steps_taken + 1
        print(
            f"[agent] step {step_number}/{self.step_limit}: requesting action",
            flush=True,
        )
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self.tools,
                reasoning_effort="medium",
                max_completion_tokens=4096,
            )
        except Exception as exc:
            print(
                f"[agent] step {step_number}: model request failed after retries "
                f"({type(exc).__name__}: {exc})",
                flush=True,
            )
            raise
        self.api_responses.append(response.model_dump(mode="json"))
        self.steps_taken += 1
        message = self.process_response(response)
        tool_names = [
            call.get("function", {}).get("name", "unknown")
            for call in message.get("tool_calls", [])
            if isinstance(call, dict)
        ]
        if tool_names:
            print(
                f"[agent] step {step_number}: tool call(s): {', '.join(tool_names)}",
                flush=True,
            )
        else:
            print(
                f"[agent] step {step_number}: response contained no parsed tool call; "
                "the loop should preserve the response and continue",
                flush=True,
            )
        return message

    def process_response(self, response: Any) -> dict[str, Any]:
        """Return relevant parts of the language model's response."""

        return response.choices[0].message.model_dump(exclude_none=True)

    def build_prompt(self) -> list[dict[str, Any]]:
        """Assemble the standing instructions, the task, and the run so far."""

        memory = (
            [
                {
                    "role": "user",
                    "content": (
                        "Working memory of the run so far, replacing the "
                        f"messages it summarizes.\n\n{self.context_summary}"
                    ),
                }
            ]
            if self.context_summary
            else []
        )
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.task_prompt},
            *memory,
            *self.history[self.compacted_history_length :],
        ]

    def estimate_active_prompt_tokens(self) -> int:
        """Estimate the next prompt, calibrated by the provider's latest usage."""

        current_prompt = self.build_prompt()
        rough_current = rough_message_tokens(current_prompt)
        if not self.api_prompts or not self.api_responses:
            return rough_current

        usage = self.api_responses[-1].get("usage") or {}
        actual_previous = usage.get("prompt_tokens")
        if not isinstance(actual_previous, int):
            return rough_current

        rough_previous = rough_message_tokens(self.api_prompts[-1])
        added_since_previous_request = max(0, rough_current - rough_previous)
        return actual_previous + added_since_previous_request

    @property
    def compaction_enabled(self) -> bool:
        """Whether this agent compacts its context at all."""

        return self.compact_threshold_tokens is not None

    def compact_context(self):
        """Replace parts of prompt with model-generated working memory. Changes the
        content that `build_prompt` emits."""

        # Keep the most recent actions whole by cutting at an assistant
        # message, so no tool observation is ever separated from the call it
        # answers. Everything older than the cut is what gets summarized.
        actions = [
            position
            for position, message in enumerate(self.history)
            if message.get("role") == "assistant"
        ]
        cut = actions[-self.compaction_keep_recent_steps]
        summarized = self.history[self.compacted_history_length : cut]

        transcript = []
        for message in summarized:
            calls = "\n".join(
                f"calls {call.get('function', {}).get('name', 'unknown')} with "
                f"{call.get('function', {}).get('arguments', '')}"
                for call in message.get("tool_calls") or []
            )
            body = "\n".join(part for part in (message.get("content"), calls) if part)
            role = message.get("role", "unknown")
            transcript.append(f"<{role}>\n{body}\n</{role}>")

        # Fold the previous memory back in, so a second compaction carries
        # forward what the first one established instead of dropping it.
        earlier = (
            [f"<working_memory>\n{self.context_summary}\n</working_memory>"]
            if self.context_summary
            else []
        )
        compaction_prompt = [
            {"role": "system", "content": COMPACTION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "\n\n".join(
                    [f"<objective>\n{self.task_prompt}\n</objective>"]
                    + earlier
                    + transcript
                ),
            },
        ]

        ### Do not modify this section ###
        compaction_response = self.client.chat.completions.create(
            model=self.model,
            messages=compaction_prompt,
            reasoning_effort="medium",
            max_completion_tokens=self.compaction_max_tokens,
        )
        ##################################

        # Only retire the summarized prefix once there is memory to stand in
        # for it; an empty summary would drop that history for nothing.
        summary = (compaction_response.choices[0].message.content or "").strip()
        if summary:
            self.context_summary = summary
            self.compacted_history_length = cut

        ### Do not modify this section ###
        return compaction_prompt, compaction_response.model_dump(mode="json")
        ##################################

    def maybe_compact_context(self) -> bool:
        """Compact before the next action request when the threshold is reached."""

        if not self.compaction_enabled:
            return False

        # Context too short to compact yet
        if self.estimate_active_prompt_tokens() < self.compact_threshold_tokens:
            return False

        prompt_before = deepcopy(self.build_prompt())

        # Not enough steps (each assistant turn corresponds to a step) to force
        # compaction yet
        if (
            len([m for m in prompt_before if m.get("role") == "assistant"])
            <= self.compaction_keep_recent_steps
        ):
            return False

        compaction_prompt, compaction_response = self.compact_context()
        prompt_after = deepcopy(self.build_prompt())
        self.compaction_events.append(
            {
                "step": self.steps_taken,
                "estimated_tokens_before": rough_message_tokens(prompt_before),
                "estimated_tokens_after": rough_message_tokens(prompt_after),
                "active_prompt_before": deepcopy(prompt_before),
                "compaction_prompt": compaction_prompt,
                "compaction_response": compaction_response,
            }
        )
        return True

    def run(self) -> None:
        """Run ReAct steps, always saving the trajectory and stopping Modal."""

        try:
            while not self.finished:
                if self.steps_taken >= self.step_limit:
                    raise StepLimitError(
                        f"Stopped after {self.steps_taken} model calls without "
                        f"finishing the task."
                    )

                self.maybe_compact_context()

                action = self.query_language_model()
                self.history.append(action)

                tool_calls = action.get("tool_calls") or []
                if not tool_calls:
                    # A reply with no tool call changed nothing, so keep it and
                    # say so rather than letting the model believe it acted.
                    self.history.append(
                        {
                            "role": "user",
                            "content": (
                                "That response made no tool call, so nothing "
                                "happened. Call a tool to act."
                            ),
                        }
                    )
                    continue

                # Executing the calls is domain-specific, and is where a
                # subclass reports that the run is over.
                self.history.extend(self.execute_tool_calls(tool_calls))
        finally:
            # This block is provided infrastructure. Do not modify it: a
            # trajectory is required even when a run fails.
            if self.logs_save_path:
                path = Path(self.logs_save_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    json.dumps(
                        {
                            "prompts": self.api_prompts,
                            "responses": self.api_responses,
                            "compactions": self.compaction_events,
                        },
                        indent=2,
                    )
                )
            if self.auto_stop_environment:
                stop = getattr(self.env, "stop", None)
                if callable(stop):
                    stop()

    def execute_tool_calls(
        self, tool_calls: list[dict[str, Any]]
    ) -> list[dict[str, str]]:
        """Execute domain-specific calls and return linked tool observations."""

        # You do not need to implement anything here. This method is
        # domain-specific and implemented by the relevant subclasses
        raise NotImplementedError
