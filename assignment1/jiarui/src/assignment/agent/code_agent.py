"""The Part 1 coding agent: fix a software issue and submit a git patch."""

from __future__ import annotations

import json
from typing import Any

from assignment.agent.base import Agent, format_tool_output
from assignment.agent.tools import EXECUTE_TOOL, FINISH_TASK_TOOL, INVOKE_SKILL_TOOL
from assignment.env import Environment
from assignment.prompts import (
    CODE_AGENT_SYSTEM_PROMPT_TEMPLATE,
    CODE_AGENT_TASK_PROMPT_TEMPLATE,
)


class CodeAgent(Agent):
    """An agent that fixes a software issue and submits a git patch."""

    def __init__(
        self,
        task: str,
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
        super().__init__(
            environment=environment,
            model=model,
            logs_save_path=logs_save_path,
            step_limit=step_limit,
            skills_path=skills_path,
            auto_stop_environment=auto_stop_environment,
            compact_threshold_tokens=compact_threshold_tokens,
            compaction_keep_recent_steps=compaction_keep_recent_steps,
            compaction_max_tokens=compaction_max_tokens,
        )
        self.task = task
        self.submitted_patch = ""
        self.system_prompt = CODE_AGENT_SYSTEM_PROMPT_TEMPLATE.render(
            system_information=json.dumps(
                {
                    "machine": environment.machine,
                    "release": environment.release,
                    "system": environment.system,
                    "version": environment.version,
                },
                indent=2,
                sort_keys=True,
            ),
            skills=self.skills,
        )
        self.task_prompt = CODE_AGENT_TASK_PROMPT_TEMPLATE.render(task=task)

        if hasattr(self, "skills") and self.skills:
            print("Using skills")
            self.tools = [EXECUTE_TOOL, FINISH_TASK_TOOL, INVOKE_SKILL_TOOL]
        else:
            print("Not using skills")
            self.tools = [EXECUTE_TOOL, FINISH_TASK_TOOL]

    def execute_tool_calls(
        self, tool_calls: list[dict[str, Any]]
    ) -> list[dict[str, str]]:
        """Execute ``execute`` and ``finish_task`` calls in the code sandbox."""

        observations = []
        for call in tool_calls:
            name = call["function"]["name"]
            try:
                arguments = json.loads(call["function"]["arguments"])
            except json.JSONDecodeError as error:
                content = f"`{name}` arguments are not valid JSON: {error}"
            else:
                if name == "execute":
                    content = format_tool_output(self.env.execute(**arguments))
                elif name == "send_message":  # FINISH_TASK_TOOL's actual name
                    content = self._submit()
                elif name == "invoke_skill":
                    content = self._invoke_skill(arguments["name"])
                else:
                    content = f"No tool named `{name}`."
            observations.append(
                {"role": "tool", "tool_call_id": call["id"], "content": content}
            )
        return observations

    def _invoke_skill(self, name: str) -> str:
        """Hand over one skill's full text, which the system prompt withholds."""

        for skill in self.skills:
            if skill["name"] == name:
                return skill["text"]
        return f"No skill named `{name}`."

    def _submit(self) -> str:
        """Accept the submission only when a non-empty patch.txt is really there."""

        result = self.env.execute("cat patch.txt")
        patch = result["output"] if result["returncode"] == 0 else ""
        if not patch.strip():
            return (
                "patch.txt is missing or empty. Write your diff there first, "
                "then submit again."
            )
        self.submitted_patch = patch
        self.finished = True
        return "Patch recorded. The run ends here."
