"""Step 2: send located before/after snippets to an LM and emit Q+A pairs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from quizzer.lm import DEFAULT_MAX_TOKENS, DEFAULT_MODEL, anthropic_client, parse_json_object, response_text
from quizzer.locate import LocateResult
from quizzer.prompts import quiz_system_prompt

DEFAULT_QUESTION_COUNT = 2


def _normalize_question_count(question_count: int) -> int:
    if not isinstance(question_count, int) or isinstance(question_count, bool) or question_count < 1:
        raise ValueError("question_count must be an integer >= 1")
    return question_count


def build_user_message(
    located: dict[str, Any], question_count: int = DEFAULT_QUESTION_COUNT
) -> str:
    count = _normalize_question_count(question_count)
    parts = [
        f"Module: {located['module']}",
        "",
        f"Write exactly {count} questions.",
        "",
        "Assignment specification for this module:",
        located.get("spec") or "(no ASSIGNMENT.md section found)",
        "",
        "Before = original unfilled stub. After = a reference solution used only "
        "to understand the intended design. Write a standardized quiz for all "
        "students on this assignment; do not refer to this after-code as anyone's "
        "particular implementation.",
        "The system prompt lists optional question strategies (motivation, "
        "alternative implementations, scenario-action matching). Use whichever "
        "fit this module; do not try to cover all of them.",
        "Write in plain, simple language, like a person writing an exam. "
        "Never use em-dashes.",
        "Quiz only this module. Ignore neighboring TODO work that happens to "
        "share a function.",
        "",
    ]
    for index, snippet in enumerate(located.get("snippets") or [], start=1):
        ids = ", ".join(snippet.get("todo_ids") or [])
        parts.extend(
            [
                f"--- snippet {index}: {snippet.get('symbol')} "
                f"({snippet.get('file')}; TODO {ids}) ---",
                "Original TODO comment:",
                "\n\n".join(snippet.get("todo_comments") or []),
                "",
                f"BEFORE {snippet['file']}:{snippet['before']['start_line']}"
                f"-{snippet['before']['end_line']}",
                snippet["before"]["code"],
                "",
                f"AFTER {snippet['file']}:{snippet['after']['start_line']}"
                f"-{snippet['after']['end_line']}",
                snippet["after"]["code"],
                "",
            ]
        )
    return "\n".join(parts).strip()


def parse_quiz_json(
    text: str, question_count: int = DEFAULT_QUESTION_COUNT
) -> dict[str, Any]:
    count = _normalize_question_count(question_count)
    payload = parse_json_object(text)
    questions = payload.get("questions")
    if not isinstance(questions, list) or not questions:
        raise ValueError("Quiz JSON must contain a non-empty `questions` list.")
    if len(questions) > count:
        payload["questions"] = questions[:count]
    return payload


def generate_quiz(
    located: dict[str, Any] | LocateResult,
    model: str = DEFAULT_MODEL,
    client: Any | None = None,
    question_count: int = DEFAULT_QUESTION_COUNT,
) -> dict[str, Any]:
    count = _normalize_question_count(question_count)
    payload = located.to_dict() if isinstance(located, LocateResult) else located
    api = client or anthropic_client()
    try:
        response = api.messages.create(
            model=model,
            max_tokens=DEFAULT_MAX_TOKENS,
            system=quiz_system_prompt(count),
            messages=[
                {"role": "user", "content": build_user_message(payload, count)},
            ],
        )
    except Exception as exc:
        raise RuntimeError(f"Language model request failed ({type(exc).__name__}): {exc}") from exc
    quiz = parse_quiz_json(response_text(response), count)
    quiz["module"] = payload.get("module")
    quiz["model"] = model
    quiz["question_count"] = count
    return quiz


def format_quiz(quiz: dict[str, Any]) -> str:
    parts = [f"# Quiz for module {quiz.get('module', '?')}", ""]
    for index, item in enumerate(quiz.get("questions") or [], start=1):
        parts.extend(
            [
                f"## Question {index} ({item.get('format', 'unknown')}, "
                f"{item.get('dimension', 'unspecified')})",
                "",
                item.get("question", "").rstrip(),
                "",
                "**Answer**",
                "",
                item.get("answer", "").rstrip(),
                "",
            ]
        )
        if item.get("explanation"):
            parts.extend(["**Why this question**", "", item["explanation"].rstrip(), ""])
    return "\n".join(parts).rstrip() + "\n"


def load_located(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
