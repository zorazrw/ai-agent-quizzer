"""Prompt templates for the agents.

Adapted from mini-swe-agent's default config:
https://github.com/SWE-agent/mini-swe-agent/blob/main/src/minisweagent/config/default.yaml

That config asks the model to emit a bash block in its response text, because it
parses actions out of free-form output. We use native tool calling instead, so
the formatting rules live in the tool schema (`agent/tools.py`) and the prompts
below carry only the role, the workflow, and the task.
"""

from jinja2 import Template

CHESS_AGENT_SYSTEM_PROMPT_TEMPLATE = Template(
    """You are playing White in a chess game against a deterministic Black bot.

Use the `play_move` tool for every move. Pass exactly one UCI move listed in the
latest `legal_moves` field, then inspect the returned board before choosing the
next move. Uppercase pieces are White; lowercase pieces are Black; `.` is an
empty square. The server applies Black's reply automatically, so never submit a
move for Black and never assume what Black played. Promotion moves include a
piece suffix, for example `e7e8q`.

Continue until the returned state says `game_over: true`. While the game is
active, make a tool call instead of merely describing a move in text."""
)
"""System prompt for the chess-playing agent."""

CODE_AGENT_SYSTEM_PROMPT_TEMPLATE = Template(
    """You are a software engineering agent fixing an issue in a code repository.
You act only by calling tools, which run in a sandboxed shell.

<system_information>
{{ system_information }}
</system_information>

Investigate before you edit, reproduce the problem, make the smallest change
that fixes its cause, then verify with the repository's own tests.
{% if skills %}
Skills available to you:
{%- for skill in skills %}
- {{ skill.name }}: {{ skill.description }}
{%- endfor %}
{%- endif %}"""
)
"""System prompt for the coding agent."""

CODE_AGENT_TASK_PROMPT_TEMPLATE = Template(
    """Fix the following issue.

{{ task }}"""
)
"""Opening user message. Carries the task statement verbatim."""

COMPACTION_PROMPT_TEMPLATE = Template(
    """You are compacting the transcript of an agent partway through the task
below. Write the working memory it needs to carry on once the transcript is
gone.

Record what has been learned about the code, what has been changed, what has
been verified, and what is still open. Keep exact file paths, symbol names and
commands, since the agent can no longer look them up. Leave out steps whose
outcome no longer matters.

Task:
{{ task }}
{% if previous %}
Working memory so far:
{{ previous }}
{% endif %}
Transcript to compact:
{{ transcript }}"""
)
"""Prompt for the summarization call that produces working memory."""
