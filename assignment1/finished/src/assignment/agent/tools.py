"""Tool definitions exposed to the model, in the OpenAI tool-calling format."""

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

SEND_MESSAGE_TOOL = {
    "type": "function",
    "function": {
        "name": "send_message",
        "description": ("Send a message to the user."),
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": ("Content of the message"),
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

PLAY_MOVE_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "play_move",
        "description": (
            "Play one move as White in the live game and return the position "
            "that results from it.\n"
            "\n"
            "Moves use UCI notation: the square the piece starts on followed by "
            "the square it ends on, as in `e2e4`. A promotion adds the promoted "
            "piece, as in `e7e8q`. Castling is written as the king's own move, "
            "as in `e1g1`.\n"
            "\n"
            "The move must be legal in the current position. The server plays "
            "Black's reply automatically, so never submit a move for Black. "
            "Play one move per turn and read the board it returns before "
            "choosing the next."
        ),
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "move": {
                    "type": "string",
                    "description": (
                        'The move to play, in UCI notation, such as "e2e4" or '
                        '"e7e8q".'
                    ),
                },
            },
            "required": ["move"],
            "additionalProperties": False,
        },
    },
}

SIMULATE_MOVE_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "simulate_move",
        "description": (
            "Examine a position, or see what one move would lead to, without "
            "touching the live game.\n"
            "\n"
            "Given a FEN alone, this returns that position and its legal moves. "
            "Given a FEN and a move, it returns the position after exactly that "
            "one ply, for whichever side is to move. Use it to look ahead "
            "before committing to a move with `play_move`.\n"
            "\n"
            "Every call starts from the FEN you supply, so nothing accumulates "
            "and there is nothing to undo. The reply reports `fen`, `squares`, "
            "`turn`, `legal_moves`, and the terminal fields `game_over`, "
            "`winner`, and `result`."
        ),
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "fen": {
                    "type": "string",
                    "description": (
                        "The position to start from, as a complete six-field "
                        'FEN, such as "7k/5Q2/6K1/8/8/8/8/8 w - - 0 1".'
                    ),
                },
                "move": {
                    "type": ["string", "null"],
                    "description": (
                        "A move to apply to that position, in UCI notation, "
                        'such as "e2e4" or "e7e8q". Pass null to inspect the '
                        "position as it stands."
                    ),
                },
            },
            "required": ["fen", "move"],
            "additionalProperties": False,
        },
    },
}

RUN_PYTHON_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "run_python",
        "description": (
            "Run a Python snippet in the sandbox next to the chess server, and "
            "return what it printed.\n"
            "\n"
            "Two functions are already defined for you. "
            "`simulate_move(fen, move=None)` returns a dict for a position, or "
            "for the position one ply after `move`, without touching the game. "
            "`play_move(move)` commits one move to the live game. Both raise on "
            "an illegal move or a bad position, so guard calls you are unsure "
            "of.\n"
            "\n"
            "Use this to search candidate moves and then commit the best one in "
            "the same snippet. Print whatever you need to see: only stdout, "
            "stderr, and an uncaught exception come back. Nothing carries over "
            "between snippets, so define everything you need each time."
        ),
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The Python source to run.",
                },
            },
            "required": ["code"],
            "additionalProperties": False,
        },
    },
}
