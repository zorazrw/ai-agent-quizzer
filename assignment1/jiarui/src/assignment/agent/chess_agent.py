"""The Part 2 chess agent: reuse the shared loop with one new domain tool."""

from __future__ import annotations

import json
from typing import Any

import httpx

from assignment.agent.base import Agent
from assignment.agent.tools import PLAY_MOVE_TOOL
from assignment.prompts import CHESS_AGENT_SYSTEM_PROMPT_TEMPLATE
from assignment.env import Environment


def format_chess_state(state: dict[str, Any]) -> str:
    """Turn chess API JSON into a compact observation an LLM can act on."""

    squares = state.get("squares", {})
    board_lines = ["    a b c d e f g h"]
    for rank in range(8, 0, -1):
        pieces = [squares.get(f"{file}{rank}", ".") for file in "abcdefgh"]
        board_lines.append(f"{rank} | {' '.join(pieces)} | {rank}")
    board_lines.append("    a b c d e f g h")

    recent_history = state.get("history", [])[-8:]
    history = (
        " ".join(
            f"{item.get('ply', '?')}:{item.get('san', '?')}" for item in recent_history
        )
        or "(none)"
    )
    legal_moves = " ".join(state.get("legal_moves", [])) or "(none)"
    board_text = "\n".join(board_lines)

    return (
        "<chess_state>\n"
        f"status: {state.get('status', 'unknown')}\n"
        f"turn: {state.get('turn', 'unknown')}\n"
        f"in_check: {state.get('in_check', False)}\n"
        f"game_over: {state.get('game_over', False)}\n"
        f"fen: {state.get('fen', '')}\n"
        f"human_move: {state.get('human_move') or '(none)'}\n"
        f"engine_move: {state.get('engine_move') or '(none)'}\n"
        "board:\n"
        f"{board_text}\n"
        f"recent_history: {history}\n"
        f"legal_moves: {legal_moves}\n"
        "</chess_state>"
    )


class ChessAgent(Agent):
    """An agent that plays White against the server's deterministic Black bot."""

    def __init__(
        self,
        environment: Environment,
        model: str | None = None,
        logs_save_path: str | None = None,
        step_limit: int = 200,
        auto_stop_environment: bool = True,
        http_client: Any | None = None,
        reset_game: bool = True,
        # None means "use the shared default", which Agent fills in.
        compact_threshold_tokens: int | None = None,
        compaction_keep_recent_steps: int | None = None,
        compaction_max_tokens: int | None = None,
    ):
        super().__init__(
            environment=environment,
            model=model,
            logs_save_path=logs_save_path,
            step_limit=step_limit,
            auto_stop_environment=auto_stop_environment,
            compact_threshold_tokens=compact_threshold_tokens,
            compaction_keep_recent_steps=compaction_keep_recent_steps,
            compaction_max_tokens=compaction_max_tokens,
        )

        self.tools: list[dict[str, Any]] = [PLAY_MOVE_TOOL]

        if http_client is None:
            server_url = getattr(environment, "server_url", "")
            if not server_url:
                raise ValueError(
                    "ChessAgent needs an environment with server_url or an http_client."
                )
            http_client = httpx.Client(base_url=server_url, timeout=20)
        self.chess_client = http_client

        method, endpoint = (
            ("POST", "/api/reset") if reset_game else ("GET", "/api/state")
        )
        initial_state = self._request_state(method, endpoint)
        self.last_state = initial_state
        self.finished = bool(initial_state.get("game_over"))
        self.system_prompt = CHESS_AGENT_SYSTEM_PROMPT_TEMPLATE.render()
        self.task_prompt = (
            "Play this game as White. Choose one move from legal_moves and "
            "call play_move.\n\n"
            f"{format_chess_state(initial_state)}"
        )

    def _request_state(
        self, method: str, endpoint: str, **kwargs: Any
    ) -> dict[str, Any]:
        """Make one chess API request and validate its JSON response."""

        response = self.chess_client.request(method, endpoint, **kwargs)
        try:
            payload = response.json()
        except ValueError as exc:
            raise RuntimeError(
                f"Chess server returned non-JSON ({response.status_code})."
            ) from exc
        if response.status_code >= 400:
            detail = (
                payload.get("detail", payload) if isinstance(payload, dict) else payload
            )
            raise ValueError(str(detail))
        if not isinstance(payload, dict):
            raise RuntimeError("Chess server response must be a JSON object.")
        return payload

    def execute_tool_calls(
        self, tool_calls: list[dict[str, Any]]
    ) -> list[dict[str, str]]:
        """Execute model-generated ``play_move`` calls against the chess API."""

        observations = []
        played = False
        for call in tool_calls:
            name = call["function"]["name"]
            if name != "play_move":
                content = f"<chess_error>No tool named `{name}`.</chess_error>"
            elif played:
                # The position changed when the first call was played, so the
                # rest were chosen against a board that no longer exists.
                content = (
                    "<chess_error>Only the first move of a turn is played. "
                    "Send one call at a time, choosing from the legal_moves in "
                    "the state above.</chess_error>"
                )
            else:
                content = self._play_move(call["function"]["arguments"])
                played = True
            observations.append(
                {"role": "tool", "tool_call_id": call["id"], "content": content}
            )
        return observations

    def _play_move(self, arguments: str) -> str:
        """Send one move and return the new position, or a recoverable error."""

        try:
            move = json.loads(arguments)["move"]
        except (ValueError, KeyError, TypeError) as error:
            return f"<chess_error>Could not read the move: {error}</chess_error>"

        try:
            state = self._request_state("POST", "/api/move", json={"move": move})
        except (ValueError, RuntimeError, httpx.HTTPError) as error:
            # An illegal move, a malformed response, or the server being
            # unreachable: all recoverable, because the game has not moved on.
            return f"<chess_error>{move} was not played: {error}</chess_error>"

        self.last_state = state
        self.finished = bool(state.get("game_over"))
        return format_chess_state(state)
