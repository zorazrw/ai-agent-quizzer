"""The Part 1 coding agent: fix a software issue and submit a git patch."""

from __future__ import annotations

import json
from typing import Any

from assignment.agent.base import (
    DEFAULT_COMPACTION_KEEP_RECENT_STEPS,
    DEFAULT_COMPACTION_MAX_TOKENS,
    Agent,
    format_tool_output,
)
from assignment.agent.tools import EXECUTE_TOOL, SEND_MESSAGE_TOOL
from assignment.env import Environment

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
        compaction_keep_recent_steps: int = DEFAULT_COMPACTION_KEEP_RECENT_STEPS,
        compaction_max_tokens: int = DEFAULT_COMPACTION_MAX_TOKENS,
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

        self.tools.extend([EXECUTE_TOOL, SEND_MESSAGE_TOOL])

        # Read from the sandbox rather than this machine, so the model plans
        # around the platform its commands actually run on.
        system_information = json.dumps(
            {
                "machine": getattr(environment, "machine", ""),
                "release": getattr(environment, "release", ""),
                "system": getattr(environment, "system", ""),
                "version": getattr(environment, "version", ""),
            },
            indent=2,
            sort_keys=True,
        )
        working_directory = getattr(environment, "cwd", "/")

        self.system_prompt = (
            "You are a software engineering agent working alone in a sandboxed "
            "terminal. Carry out the task you are given by inspecting the "
            "environment, running commands, and checking the results yourself. "
            "No one can answer questions: decide and continue.\n"
            "\n"
            f"<system_information>\n{system_information}\n</system_information>\n"
            "\n"
            f"Commands run in {working_directory} unless the call says "
            "otherwise, each in a fresh subshell: a `cd` or an export does not "
            "carry over, though written files persist. Nothing can prompt for "
            "input, so pass flags like `-y`. Read files in slices; long output "
            "is truncated before you see it.\n"
            "\n"
            "Look before you act, change one thing at a time, and confirm what "
            "happened from the command's own output instead of assuming it. "
            "Keep changes as narrow as the task calls for and follow the "
            "conventions already in the files you touch. When something fails, "
            "read the error and adapt rather than repeating the command.\n"
            "\n"
            "Each turn, call a tool: described actions do not happen. Once the "
            "task is done and you have checked it, call `send_message` with a "
            "summary of what you did and how you verified it. That ends the run."
        )

        self.task_prompt = f"Complete this task.\n\n<task>\n{task}\n</task>"

        # Advertise what each skill covers, but not how it works: the body
        # arrives only when the agent asks for it with `invoke_skill`.
        if self.skills:
            catalog = "\n\n".join(skill["metadata"] for skill in self.skills.values())
            self.system_prompt += (
                "\n\nReusable skills are available. Call `invoke_skill` with a "
                "skill's name to load its instructions, and follow them in place "
                "of your default approach. Check this list before you start and "
                f"again before you finish.\n\n<skills>\n{catalog}\n</skills>\n"
            )

    def execute_tool_calls(
        self, tool_calls: list[dict[str, Any]]
    ) -> list[dict[str, str]]:
        """Execute ``execute`` and ``send_message`` calls in the code sandbox."""

        observations: list[dict[str, str]] = []
        for call in tool_calls:
            function = call.get("function") or {}
            name = function.get("name", "")

            # Anything the model got wrong becomes an observation it can read
            # and correct on its next turn, never an exception that ends a run.
            try:
                arguments = json.loads(function.get("arguments") or "{}")
            except json.JSONDecodeError as exc:
                result = {"error": f"Arguments were not valid JSON: {exc}."}
            else:
                if not isinstance(arguments, dict):
                    result = {"error": "Arguments must be a JSON object."}
                elif name == "execute":
                    command = arguments.get("command")
                    if not isinstance(command, (str, list)):
                        result = {
                            "error": "execute needs `command` as a string or a "
                            "list of strings."
                        }
                    else:
                        # Whatever the model leaves out keeps the sandbox's
                        # default, so a bare command still runs sensibly.
                        options = {
                            key: arguments[key]
                            for key in ("timeout", "cwd", "env", "shell")
                            if key in arguments
                        }
                        output = self.env.execute(command, **options)
                        result = {
                            "output": output.get("output", ""),
                            "returncode": output.get("returncode", ""),
                        }
                        if output.get("exception_info"):
                            result["exception_info"] = output["exception_info"]
                elif name == "invoke_skill" and self.skills:
                    requested = arguments.get("name")
                    if not isinstance(requested, str):
                        result = {"error": "invoke_skill needs `name` as a string."}
                    elif requested not in self.skills:
                        result = {
                            "error": f"There is no skill named {requested!r}. "
                            f"Available skills: {', '.join(sorted(self.skills))}."
                        }
                    else:
                        result = {"skill": self.skills[requested]["content"]}
                elif name == "send_message":
                    summary = arguments.get("summary")
                    if not isinstance(summary, str):
                        result = {"error": "send_message needs `summary` as a string."}
                    else:
                        self.finished = True
                        result = {"status": "Message delivered. The run is over."}
                else:
                    result = {"error": f"There is no tool named {name!r}."}

            observations.append(
                {
                    "role": "tool",
                    "tool_call_id": call.get("id", ""),
                    "content": format_tool_output(result),
                }
            )

        return observations
