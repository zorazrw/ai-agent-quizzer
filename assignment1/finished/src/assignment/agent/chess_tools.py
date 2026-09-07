"""Chess tool implementations, decoupled from the agent that registers them.

Every function here takes the HTTP client explicitly instead of reading it off
an agent, so the same code can run in the agent process or inside the sandbox
beside the server it talks to.
"""

from __future__ import annotations

import base64
import json
from typing import Any

import httpx

CHESS_PORT = 8000


def _request_state(
    client: httpx.Client, method: str, endpoint: str, **kwargs: Any
) -> dict[str, Any]:
    """Make one chess API request and validate its JSON response."""

    response = client.request(method, endpoint, **kwargs)
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


def _simulate_move(client: httpx.Client, arguments: str) -> str:
    """New tool: inspect FEN or simulate one ply without changing the game.

    Takes the raw JSON arguments of one tool call and returns the observation
    to send back, so a bad argument or a server error reaches the model as a
    recoverable ``<chess_error>`` instead of ending the run.
    """
    try:
        parsed = json.loads(arguments or "{}")
    except json.JSONDecodeError as exc:
        return f"<chess_error>Arguments were not valid JSON: {exc}.</chess_error>"

    if not isinstance(parsed, dict):
        return "<chess_error>Arguments must be a JSON object.</chess_error>"

    fen = parsed.get("fen")
    if not isinstance(fen, str):
        return (
            "<chess_error>simulate_move needs `fen` as a string holding a "
            "complete six-field FEN.</chess_error>"
        )

    # A null or absent move asks for the position itself rather than the
    # position after a ply, so it is not an error.
    move = parsed.get("move")
    if move is not None and not isinstance(move, str):
        return (
            "<chess_error>simulate_move needs `move` as a string in UCI "
            "notation, such as e2e4, or null to inspect the position."
            "</chess_error>"
        )

    payload: dict[str, Any] = {"fen": fen}
    if move is not None:
        payload["move"] = move

    try:
        state = _request_state(client, "POST", "/api/simulate", json=payload)
    except httpx.HTTPError as exc:
        return f"<chess_error>Could not reach the chess server: {exc}</chess_error>"
    except (ValueError, RuntimeError) as exc:
        return f"<chess_error>{exc}</chess_error>"

    return json.dumps(state)


def _play_move(client: httpx.Client, arguments: str) -> str:
    """Existing tool: play one move as White and return the resulting state.

    Takes the raw JSON arguments of one tool call. Returns the new state, or a
    `<chess_error>` observation if the move could not be played.
    """
    try:
        parsed = json.loads(arguments or "{}")
    except json.JSONDecodeError as exc:
        return f"<chess_error>Arguments were not valid JSON: {exc}.</chess_error>"

    if not isinstance(parsed, dict):
        return "<chess_error>Arguments must be a JSON object.</chess_error>"

    move = parsed.get("move")
    if not isinstance(move, str):
        return (
            "<chess_error>play_move needs `move` as a string in UCI notation, "
            "such as e2e4.</chess_error>"
        )

    try:
        state = _request_state(client, "POST", "/api/move", json={"move": move})
    except httpx.HTTPError as exc:
        return f"<chess_error>Could not reach the chess server: {exc}</chess_error>"
    except (ValueError, RuntimeError) as exc:
        return f"<chess_error>{exc}</chess_error>"

    return json.dumps(state)


def _run_python(env: Any, port: int, arguments: str) -> str:
    """New tool: run Python with access to the existing registered tools.

    The snippet runs inside the sandbox, which already has the tool
    implementations and the chess server, so code the model wrote never
    executes in the agent process.
    """
    try:
        parsed = json.loads(arguments or "{}")
    except json.JSONDecodeError as exc:
        return f"<chess_error>Arguments were not valid JSON: {exc}.</chess_error>"

    if not isinstance(parsed, dict):
        return "<chess_error>Arguments must be a JSON object.</chess_error>"

    code = parsed.get("code")
    if not isinstance(code, str):
        return "<chess_error>run_python needs `code` as a string.</chess_error>"

    # Base64 so no quoting in the snippet can break out of the command line.
    encoded = base64.b64encode(code.encode()).decode()
    result = env.execute(f"python /opt/assignment/sandbox_python.py {port} {encoded}")

    # A non-zero exit is the runner itself failing. An exception raised by the
    # snippet exits zero and is reported in the JSON, as the model's problem.
    if result.get("returncode") != 0:
        detail = (
            result.get("exception_info")
            or result.get("stderr")
            or result.get("output")
            or "the runner produced no output"
        )
        return (
            f"<chess_error>Could not run the snippet: {str(detail).strip()}"
            "</chess_error>"
        )

    # The runner prints its JSON to the real stdout, keeping it clear of
    # anything the snippet itself printed.
    printed = result.get("stdout")
    return printed if printed is not None else result.get("output", "")


def _invoke_skill(skills: dict[str, dict[str, str]], arguments: str) -> str:
    """Existing tool: load one skill's instructions into the conversation."""
    try:
        parsed = json.loads(arguments or "{}")
    except json.JSONDecodeError as exc:
        return f"<chess_error>Arguments were not valid JSON: {exc}.</chess_error>"

    if not isinstance(parsed, dict):
        return "<chess_error>Arguments must be a JSON object.</chess_error>"

    name = parsed.get("name")
    if not isinstance(name, str):
        return "<chess_error>invoke_skill needs `name` as a string.</chess_error>"

    if name not in skills:
        available = ", ".join(sorted(skills)) or "none"
        return (
            f"<chess_error>There is no skill named {name!r}. Available skills: "
            f"{available}.</chess_error>"
        )

    return skills[name]["content"]


def _game_state(client: httpx.Client, reset: bool = False) -> dict:
    """Read the live game, or start a new one and read the opening position."""

    method, endpoint = ("POST", "/api/reset") if reset else ("GET", "/api/state")
    return _request_state(client, method, endpoint)
