"""Tests for quizzer.quiz helpers. LM calls are stubbed; nothing is billed."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from quizzer.lm import anthropic_client, response_text
from quizzer.locate import LocateResult, Snippet, Span
from quizzer.prompts import quiz_system_prompt
from quizzer.quiz import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_QUESTION_COUNT,
    build_user_message,
    format_quiz,
    generate_quiz,
    load_located,
    parse_quiz_json,
)

LOCATED = {
    "module": "1.1",
    "spec": "Build the prompt.",
    "snippets": [
        {
            "todo_ids": ["1.1.a"],
            "todo_comments": ["# TODO(1.1.a): history"],
            "file": "src/agent.py",
            "symbol": "Agent.build_prompt",
            "before": {"start_line": 1, "end_line": 3, "code": "raise NotImplementedError\n"},
            "after": {
                "start_line": 1,
                "end_line": 4,
                "code": "return [system, user, *history]\n",
            },
        }
    ],
}

QUESTIONS = {
    "questions": [
        {
            "format": "mcq",
            "dimension": "design",
            "question": "Why keep history separate?\nA. x\nB. y\nC. z\nD. w",
            "answer": "B. So standing instructions stay verbatim.",
            "explanation": "Compaction should not rewrite the task.",
        }
    ]
}


class FakeMessages:
    """Record ``messages.create`` kwargs and return canned text or raise."""

    def __init__(self, content: str | None, error: Exception | None = None):
        """Remember canned reply text or an error to raise on create."""
        self.content = content
        self.error = error
        self.calls: list[dict] = []

    def create(self, **kwargs):
        """Store the request, then raise or wrap ``content`` as a Messages response."""
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return SimpleNamespace(content=[SimpleNamespace(text=self.content)])


class FakeClient:
    """Stand-in Anthropic client whose ``messages.create`` is a FakeMessages."""

    def __init__(self, content: str | None = json.dumps(QUESTIONS), error: Exception | None = None):
        """Attach a FakeMessages that returns ``content`` or ``error``."""
        self.messages = FakeMessages(content, error)


def test_build_user_message_includes_spec_and_spans():
    """The quiz prompt includes the module spec, TODO comment, and before/after spans."""
    text = build_user_message(LOCATED)
    assert "Module: 1.1" in text
    assert "Build the prompt." in text
    assert "Agent.build_prompt" in text
    assert "TODO(1.1.a): history" in text
    assert "BEFORE src/agent.py:1-3" in text
    assert "raise NotImplementedError" in text
    assert "AFTER src/agent.py:1-4" in text
    assert "return [system, user, *history]" in text
    assert "Quiz only this module" in text
    assert "reference solution" in text
    assert "standardized quiz" in text
    assert "optional question strategies" in text
    assert "do not try to cover all of them" in text
    assert "Never use em-dashes." in text
    assert f"Write exactly {DEFAULT_QUESTION_COUNT} questions." in text


def test_build_user_message_uses_requested_question_count():
    """The user prompt names the requested number of questions."""
    text = build_user_message(LOCATED, question_count=4)
    assert "Write exactly 4 questions." in text


def test_build_user_message_handles_empty_snippets_and_spec():
    """Missing spec and snippets still produce a prompt with a fallback note."""
    text = build_user_message({"module": "1.2"})
    assert "Module: 1.2" in text
    assert "no ASSIGNMENT.md section found" in text
    assert "BEFORE" not in text


def test_parse_quiz_json_raw_and_fenced():
    """Bare JSON and markdown-fenced JSON both parse into the questions payload."""
    raw = parse_quiz_json(json.dumps(QUESTIONS))
    assert raw["questions"][0]["format"] == "mcq"
    fenced = parse_quiz_json("Here:\n```json\n" + json.dumps(QUESTIONS) + "\n```\n")
    assert fenced["questions"][0]["dimension"] == "design"
    untagged = parse_quiz_json("```\n" + json.dumps(QUESTIONS) + "\n```")
    assert untagged["questions"][0]["format"] == "mcq"


def test_parse_quiz_json_truncates_to_requested_count():
    """Extra questions are cut down to the requested count."""
    payload = {
        "questions": [
            {"format": "mcq", "question": "q1", "answer": "a1"},
            {"format": "short_answer", "question": "q2", "answer": "a2"},
            {"format": "mcq", "question": "q3", "answer": "a3"},
        ]
    }
    parsed = parse_quiz_json(json.dumps(payload))
    assert len(parsed["questions"]) == 2
    assert parsed["questions"][0]["question"] == "q1"
    assert parsed["questions"][1]["question"] == "q2"
    keep_three = parse_quiz_json(json.dumps(payload), question_count=3)
    assert [item["question"] for item in keep_three["questions"]] == ["q1", "q2", "q3"]
    keep_one = parse_quiz_json(json.dumps(payload), question_count=1)
    assert [item["question"] for item in keep_one["questions"]] == ["q1"]


def test_parse_quiz_json_rejects_invalid_and_empty():
    """Non-JSON, non-objects, missing/empty/non-list questions are rejected."""
    with pytest.raises(ValueError, match="did not return JSON"):
        parse_quiz_json("not json")
    with pytest.raises(ValueError, match="must be an object"):
        parse_quiz_json("[1]")
    with pytest.raises(ValueError, match="non-empty"):
        parse_quiz_json("{}")
    with pytest.raises(ValueError, match="non-empty"):
        parse_quiz_json('{"questions": []}')
    with pytest.raises(ValueError, match="non-empty"):
        parse_quiz_json('{"questions": "nope"}')


def test_generate_quiz_from_dict_and_locate_result():
    """generate_quiz accepts a dict or LocateResult and sends the quiz system prompt."""
    client = FakeClient()
    quiz = generate_quiz(LOCATED, model="claude-sonnet-4-6", client=client)
    assert quiz["module"] == "1.1"
    assert quiz["model"] == "claude-sonnet-4-6"
    assert quiz["questions"][0]["answer"].startswith("B.")
    assert quiz["question_count"] == DEFAULT_QUESTION_COUNT
    call = client.messages.calls[0]
    assert call["model"] == "claude-sonnet-4-6"
    assert call["max_tokens"] == DEFAULT_MAX_TOKENS
    assert call["system"] == quiz_system_prompt(DEFAULT_QUESTION_COUNT)
    assert call["messages"] == [
        {"role": "user", "content": build_user_message(LOCATED, DEFAULT_QUESTION_COUNT)},
    ]
    assert all(message["role"] != "system" for message in call["messages"])

    default_client = FakeClient()
    generate_quiz(LOCATED, client=default_client)
    assert default_client.messages.calls[0]["model"] == DEFAULT_MODEL

    result = LocateResult(
        module="1.1",
        finished_dir="/f",
        original_dir="/o",
        spec="Build the prompt.",
        snippets=[
            Snippet(
                todo_ids=["1.1.a"],
                todo_comments=["# TODO(1.1.a): history"],
                file="src/agent.py",
                symbol="Agent.build_prompt",
                before=Span(1, 3, "raise NotImplementedError\n"),
                after=Span(1, 4, "return [system, user, *history]\n"),
            )
        ],
    )
    result_client = FakeClient()
    quiz2 = generate_quiz(result, client=result_client)
    assert quiz2["module"] == "1.1"
    user_content = result_client.messages.calls[0]["messages"][0]["content"]
    assert "Build the prompt." in user_content
    assert "Agent.build_prompt" in user_content


def test_generate_quiz_honors_question_count():
    """The model is asked for the requested number of questions, and extras are trimmed."""
    payload = {
        "questions": [
            {"format": "mcq", "question": "q1", "answer": "a1"},
            {"format": "short_answer", "question": "q2", "answer": "a2"},
            {"format": "mcq", "question": "q3", "answer": "a3"},
        ]
    }
    client = FakeClient(content=json.dumps(payload))
    quiz = generate_quiz(LOCATED, client=client, question_count=3)
    assert quiz["question_count"] == 3
    assert len(quiz["questions"]) == 3
    assert "exactly 3 questions" in client.messages.calls[0]["system"]
    assert "Write exactly 3 questions." in client.messages.calls[0]["messages"][0]["content"]

    trimmed = generate_quiz(
        LOCATED, client=FakeClient(content=json.dumps(payload)), question_count=1
    )
    assert [item["question"] for item in trimmed["questions"]] == ["q1"]


def test_generate_quiz_rejects_invalid_question_count():
    """Zero or negative question counts are rejected before calling the model."""
    with pytest.raises(ValueError, match="question_count"):
        generate_quiz(LOCATED, client=FakeClient(), question_count=0)


def test_generate_quiz_wraps_client_errors_and_empty_content():
    """API failures become RuntimeError; empty model text fails JSON parse."""
    with pytest.raises(RuntimeError, match="Language model request failed"):
        generate_quiz(LOCATED, client=FakeClient(error=RuntimeError("timeout")))
    with pytest.raises(ValueError, match="did not return JSON"):
        generate_quiz(LOCATED, client=FakeClient(content=None))


def test_format_quiz_with_and_without_explanation():
    """Markdown includes explanation when present and uses fallbacks when fields are missing."""
    rendered = format_quiz({"module": "1.1", "questions": QUESTIONS["questions"]})
    assert "# Quiz for module 1.1" in rendered
    assert "## Question 1 (mcq, design)" in rendered
    assert "**Answer**" in rendered
    assert "**Why this question**" in rendered
    assert "Compaction should not rewrite the task." in rendered

    bare = format_quiz(
        {
            "questions": [
                {"question": "Why?", "answer": "Because."}
            ]
        }
    )
    assert "# Quiz for module ?" in bare
    assert "unknown" in bare
    assert "unspecified" in bare
    assert "**Why this question**" not in bare


def test_load_located(tmp_path: Path):
    """load_located reads locate JSON from disk."""
    path = tmp_path / "loc.json"
    path.write_text(json.dumps(LOCATED), encoding="utf-8")
    assert load_located(path)["module"] == "1.1"


def test_response_text_joins_text_blocks():
    """Text blocks are concatenated; tool-use blocks and missing content are skipped."""
    response = SimpleNamespace(
        content=[
            SimpleNamespace(text="hello "),
            SimpleNamespace(type="tool_use"),
            SimpleNamespace(text="world"),
        ]
    )
    assert response_text(response) == "hello world"
    assert response_text(SimpleNamespace(content=[])) == ""
    assert response_text(SimpleNamespace()) == ""


def test_client_requires_anthropic_key(monkeypatch):
    """Missing or placeholder ANTHROPIC_API_KEY is rejected."""
    monkeypatch.setattr("quizzer.lm.load_dotenv", lambda: None)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        anthropic_client()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "replace-with-your-key")
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        anthropic_client()


def test_client_passes_anthropic_key(monkeypatch):
    """A real-looking key is forwarded to the Anthropic constructor."""
    monkeypatch.setattr("quizzer.lm.load_dotenv", lambda: None)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    captured: dict = {}

    def fake_anthropic(**kwargs):
        """Capture constructor kwargs instead of talking to the network."""
        captured.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr("quizzer.lm.Anthropic", fake_anthropic)
    anthropic_client()
    assert captured == {"api_key": "sk-ant-test"}


def test_quiz_system_prompt_states_limits():
    """The quiz prompt caps length, requires JSON, and lists question strategies."""
    prompt = quiz_system_prompt(DEFAULT_QUESTION_COUNT)
    assert f"exactly {DEFAULT_QUESTION_COUNT} questions" in prompt
    assert "Return only a JSON object" in prompt
    assert "Quiz only this module" in prompt
    assert "Question strategies" in prompt
    assert "### 1. Motivation" in prompt
    assert "### 2. Alternative implementations" in prompt
    assert "### 3. Scenario-action matching" in prompt
    assert "optional tools, not a checklist" in prompt
    assert "Do not try to cover all three" in prompt
    assert "exactly 4 options" in prompt
    assert "At least 4 situations" in prompt
    assert "conceptual understanding" in prompt
    assert "plain, simple language" in prompt
    assert "Never use em-dashes" in prompt
    assert "standardized" in prompt.lower()
    assert "private reference" in prompt
    assert "as the student wrote" in prompt
    assert "exactly 5 questions" in quiz_system_prompt(5)
