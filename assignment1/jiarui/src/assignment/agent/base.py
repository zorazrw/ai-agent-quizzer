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

from dotenv import load_dotenv
from openai import OpenAI

from assignment.env import Environment
from assignment.prompts import COMPACTION_PROMPT_TEMPLATE

load_dotenv()
logger = logging.getLogger(__name__)

# Observations longer than this are truncated before they reach the model.
# Used by format_tool_output below, which is provided, so it lives outside the
# solution block.
MAX_OBSERVATION_CHARS = 10_000


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


def estimate_tokens(messages: list[dict[str, Any]]) -> int:
    """Estimate a message list's token count, at roughly four characters each."""

    return len(json.dumps(messages)) // 4


def load_skills(skills_path: str | None) -> list[dict[str, str]]:
    """Read every skill under ``skills_path``.

    Name and description are kept apart from the full text so a prompt can
    announce a skill without disclosing what it says.
    """

    if not skills_path:
        return []

    skills = []
    for skill_file in sorted(Path(skills_path).glob("*/SKILL.md")):
        text = skill_file.read_text()
        fields = {}
        for line in text.split("---")[1].strip().splitlines():
            key, _, value = line.partition(":")
            fields[key.strip()] = value.strip()
        skills.append(
            {"name": fields["name"], "description": fields["description"], "text": text}
        )
    return skills


class Agent:
    """A domain-independent ReAct agent with pluggable tools."""

    def __init__(
        self,
        environment: Environment,
        model: str | None = None,
        logs_save_path: str | None = None,
        step_limit: int = 100,
        skills_path: str | None = None,
        auto_stop_environment: bool = True,
        compact_threshold_tokens: int | None = None,
        compaction_keep_recent_steps: int = None,
        compaction_max_tokens: int = None,
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

        # Debugging aid: AGENT_HTTP_TRACE=<path> records every request and
        # response, retries included. Off unless the variable is set.
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )

        self.logs_save_path = logs_save_path
        self.step_limit = step_limit
        self.auto_stop_environment = auto_stop_environment
        self.compact_threshold_tokens = compact_threshold_tokens
        self.compaction_keep_recent_steps = compaction_keep_recent_steps
        self.compaction_max_tokens = compaction_max_tokens

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
        # The actions taken and observations seen so far, replayed by
        # build_prompt. Nothing else stores them: api_prompts and api_responses
        # are the saved log, not the live conversation.
        self.history: list[dict[str, Any]] = []
        self.skills = load_skills(skills_path)
        # Replaces the history that compaction removes.
        self.working_memory: str = ""

    def query_language_model(self) -> dict[str, Any]:
        """Send one tool-enabled Chat Completions request and normalize it."""

        messages = self.build_prompt()
        self.api_prompts.append(deepcopy(messages))
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=self.tools,
            reasoning_effort="low",
            max_completion_tokens=4096,
        )
        self.api_responses.append(response.model_dump(mode="json"))
        self.steps_taken += 1
        return self.process_response(response)

    def process_response(self, response: Any) -> dict[str, Any]:
        """Return the assistant message in a form reusable as prompt history."""

        return response.choices[0].message.model_dump(exclude_none=True)

    def build_prompt(self) -> list[dict[str, Any]]:
        """Build the next model prompt from ``system_prompt`` and ``task_prompt``.

        Every agent starts from the same two messages; what an implementation
        adds after them is what lets the model see its own earlier steps.
        """

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.task_prompt},
        ]
        if self.working_memory:
            messages.append({"role": "user", "content": self.working_memory})
        messages.extend(self.history)
        return messages

    def run(self) -> None:
        """Run ReAct steps, always saving the trajectory and stopping Modal."""

        try:
            while not self.finished:
                if self.steps_taken >= self.step_limit:
                    raise StepLimitError(f"Unfinished after {self.step_limit} steps.")
                self.compact_context()
                action = self.query_language_model()

                tool_calls = action.get("tool_calls", [])
                if not tool_calls:
                    # Nothing to execute, so this reply cannot be answered by a
                    # tool message. Dropping it keeps the history from holding
                    # two assistant messages in a row, which the API rejects as
                    # an invalid request once a few of them accumulate.
                    continue

                self.history.append(action)
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

    def history_steps(self) -> list[list[dict[str, Any]]]:
        """Group the history into steps: one action and the tool messages for it."""

        steps: list[list[dict[str, Any]]] = []
        for message in self.history:
            if message["role"] == "assistant":
                steps.append([message])
            elif steps:
                steps[-1].append(message)
        return steps

    def compact_context(self) -> None:
        """Summarize older steps into working memory once the prompt is large."""

        if not self.compact_threshold_tokens:
            return
        before = estimate_tokens(self.build_prompt())
        if before < self.compact_threshold_tokens:
            return

        steps = self.history_steps()
        keep = self.compaction_keep_recent_steps or 1
        if len(steps) <= keep:
            # Every step is one the most recent steps must keep, so there is
            # nothing older to summarize.
            return
        older, recent = steps[:-keep], steps[-keep:]

        transcript = []
        for step in older:
            for message in step:
                for call in message.get("tool_calls") or []:
                    function = call["function"]
                    transcript.append(
                        f"action: {function['name']}({function['arguments']})"
                    )
                if message.get("content"):
                    label = "said" if message["role"] == "assistant" else "observation"
                    transcript.append(f"{label}: {message['content']}")

        messages = [
            {
                "role": "user",
                "content": COMPACTION_PROMPT_TEMPLATE.render(
                    task=self.task_prompt,
                    previous=self.working_memory,
                    transcript="\n".join(transcript),
                ),
            }
        ]
        # No tools: this call summarizes, it does not act.
        arguments: dict[str, Any] = {"model": self.model, "messages": messages}
        if self.compaction_max_tokens:
            arguments["max_completion_tokens"] = self.compaction_max_tokens
        response = self.client.chat.completions.create(**arguments)

        self.working_memory = response.choices[0].message.content or ""
        self.history = [message for step in recent for message in step]

        self.compaction_events.append(
            {
                "prompt": messages,
                "response": response.model_dump(mode="json"),
                "step": self.steps_taken,
                "estimated_tokens_before": before,
                "estimated_tokens_after": estimate_tokens(self.build_prompt()),
            }
        )

    def execute_tool_calls(
        self, tool_calls: list[dict[str, Any]]
    ) -> list[dict[str, str]]:
        """Execute domain-specific calls and return linked tool observations."""

        raise NotImplementedError
