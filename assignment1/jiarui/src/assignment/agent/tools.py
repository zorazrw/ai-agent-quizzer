"""Tool definitions exposed to the model, in the OpenAI tool-calling format.

The command guidance below is adapted from the "Important Rules" and "Useful
command examples" of mini-swe-agent's default config:
https://github.com/SWE-agent/mini-swe-agent/blob/main/src/minisweagent/config/default.yaml
It lives in the tool description rather than the system prompt because it
describes how this tool behaves, not how the agent should approach a task.
"""

EXECUTE_TOOL = {
    "type": "function",
    "function": {
        "name": "execute",
        "description": (
            "Run a bash command and return its stdout, stderr, and exit code. "
            "A non-zero exit code is reported, not raised.\n"
            "\n"
            "Every command runs in a new subshell, so a `cd` or an export does not "
            "carry over to the next command. Use the `cwd` and `env` arguments "
            "instead. Files you write do persist.\n"
            "\n"
            "Commands are non-interactive and cannot prompt for input, so pass "
            "flags like `-y` where a command would otherwise ask for confirmation. "
            "Prefer commands that produce little output; when reading a file, use "
            "`head`, `tail`, or `sed -n '10,20p'` rather than printing all of it.\n"
            "\n"
            "Useful patterns:\n"
            "- Create a file: `cat <<'EOF' > newfile.py` ... `EOF`\n"
            "- Edit in place: `sed -i 's/old/new/g' filename.py` (drop the trailing "
            "`g` to replace only the first match; restrict to a line range with "
            "`sed -i '1,10s/old/new/g'`)\n"
            "- View numbered lines: `nl -ba filename.py | sed -n '10,20p'`"
        ),
        # The nested env object intentionally accepts arbitrary variable names,
        # which is incompatible with strict schemas on some providers.
        "strict": False,
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "anyOf": [
                        {
                            "type": "string",
                            "description": 'A shell command line, e.g. "ls -la | head".',
                        },
                        {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                'The command as an argv list, e.g. ["ls", "-la"]. '
                                "Use this with shell=false when arguments contain "
                                "characters the shell would interpret."
                            ),
                        },
                    ],
                    "description": "The command to run.",
                },
                "shell": {
                    "type": ["boolean", "null"],
                    "description": (
                        "Whether to run the command through a shell, which enables "
                        "pipes, redirection, and globbing. Defaults to true. Set to "
                        "false when passing an argv list."
                    ),
                },
                "cwd": {
                    "type": ["string", "null"],
                    "description": (
                        "Absolute path to run the command in. Defaults to the "
                        "sandbox's current working directory."
                    ),
                },
                "timeout": {
                    "type": ["number", "null"],
                    "description": (
                        "Seconds to allow the command to run before killing it. "
                        "Defaults to no timeout."
                    ),
                },
                "env": {
                    "type": ["object", "null"],
                    "additionalProperties": {"type": "string"},
                    "description": "Extra environment variables to set for this command.",
                },
            },
            "required": ["command"],
            "additionalProperties": False,
        },
    },
}

FINISH_TASK_TOOL = {
    "type": "function",
    "function": {
        "name": "send_message",
        "description": (
            # "Signal that the task is complete and end the run. Call this once you "
            # "have verified your work, and only then; no further commands will run "
            # "afterwards. Use the summary to report what you did and, if the task "
            # "asked a question, to give the answer.\n"
            "Send a message to the user."
        ),
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": (
                        "Content of the message"
                        # "A summary of what was accomplished, including the answer "
                        # "if the task called for one."
                    ),
                },
            },
            "required": ["summary"],
            "additionalProperties": False,
        },
    },
}

INVOKE_SKILL_TOOL = {
    "type": "function",
    "function": {
        "name": "invoke_skill",
        "description": (
            "Load a skill and return its instructions. A skill is a short guide "
            "for one kind of work, written ahead of time.\n"
            "\n"
            "Call this before starting work a skill covers, and follow what it "
            "says in place of your default approach."
        ),
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": (
                        "The skill's directory name, for example `hello-skill`."
                    ),
                },
            },
            "required": ["name"],
            "additionalProperties": False,
        },
    },
}

PLAY_MOVE_TOOL = {
    "type": "function",
    "function": {
        "name": "play_move",
        "description": (
            "Play one move for White and return the resulting position.\n"
            "\n"
            "Moves use UCI notation: the square the piece leaves followed by the "
            "square it lands on, for example `e2e4`. A promotion adds the new "
            "piece, for example `e7e8q`. Castling is written as the king's own "
            "move, for example `e1g1`.\n"
            "\n"
            "The server plays Black's reply itself, so the returned state "
            "already includes it."
        ),
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "move": {
                    "type": "string",
                    "description": "The move in UCI notation, for example `e2e4`.",
                },
            },
            "required": ["move"],
            "additionalProperties": False,
        },
    },
}


TOOLS = [EXECUTE_TOOL, FINISH_TASK_TOOL]
