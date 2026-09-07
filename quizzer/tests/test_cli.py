"""Tests for `python -m quizzer` commands."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from quizzer import __main__ as cli
from quizzer.locate import (
    DEFAULT_FINISHED_DIR,
    DEFAULT_ORIGINAL_DIR,
    LocateResult,
    Snippet,
    Span,
)
from quizzer.quiz import DEFAULT_QUESTION_COUNT
from quizzer.tests.helpers import write_tree

FAKE_RESULT = LocateResult(
    module="1.1",
    finished_dir="/finished",
    original_dir="/original",
    spec="Build the prompt.",
    snippets=[
        Snippet(
            todo_ids=["1.1.a"],
            todo_comments=["# TODO(1.1.a)"],
            file="src/agent.py",
            symbol="Agent.build_prompt",
            before=Span(1, 2, "raise NotImplementedError\n"),
            after=Span(1, 3, "return messages\n"),
        )
    ],
)

FAKE_QUIZ = {
    "module": "1.1",
    "model": "claude-sonnet-4-6",
    "questions": [
        {
            "format": "short_answer",
            "dimension": "design",
            "question": "Why this shape?",
            "answer": "So subclasses share the loop.",
        }
    ],
}


def test_build_parser_requires_a_subcommand():
    """A subcommand is required; locate/quiz/run flags parse, including --questions."""
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])
    args = parser.parse_args(["locate", "1.1", "--format", "json"])
    assert args.command == "locate"
    assert args.module == "1.1"
    assert args.finished == DEFAULT_FINISHED_DIR
    assert args.original == DEFAULT_ORIGINAL_DIR
    assert args.no_lm_filter is False
    assert args.assignment is None
    filtered = parser.parse_args(
        ["locate", "1.1", "--assignment", "ASSIGNMENT.md", "--no-lm-filter"]
    )
    assert filtered.assignment == Path("ASSIGNMENT.md")
    assert filtered.no_lm_filter is True
    quiz_args = parser.parse_args(["quiz", "loc.json", "-n", "4"])
    assert quiz_args.questions == 4
    run_args = parser.parse_args(["run", "1.1", "--questions", "1"])
    assert run_args.questions == 1
    assert run_args.finished == DEFAULT_FINISHED_DIR
    assert run_args.assignment is None
    quiz_default = parser.parse_args(["quiz", "loc.json"])
    assert quiz_default.questions == DEFAULT_QUESTION_COUNT
    testing_args = parser.parse_args(["testing", "--tests", "tests"])
    assert testing_args.command == "testing"
    assert testing_args.finished == DEFAULT_FINISHED_DIR
    assert testing_args.original == DEFAULT_ORIGINAL_DIR
    assert testing_args.tests == Path("tests")
    with pytest.raises(SystemExit):
        parser.parse_args(["quiz", "loc.json", "--questions", "0"])


def test_main_locate_json_and_out(monkeypatch, capsys, tmp_path: Path):
    """locate --format json prints JSON and writes the same payload to -o."""
    monkeypatch.setattr(cli, "locate_module", lambda *args, **kwargs: FAKE_RESULT)
    out = tmp_path / "loc.json"
    assert cli.main(["locate", "1.1", "--format", "json", "-o", str(out)]) == 0
    printed = json.loads(capsys.readouterr().out)
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert printed["module"] == "1.1"
    assert saved["snippets"][0]["symbol"] == "Agent.build_prompt"


def test_main_locate_markdown(monkeypatch, capsys):
    """Default locate output is the markdown dump."""
    monkeypatch.setattr(cli, "locate_module", lambda *args, **kwargs: FAKE_RESULT)
    assert cli.main(["locate", "1.1"]) == 0
    output = capsys.readouterr().out
    assert "# Module 1.1" in output
    assert "Agent.build_prompt" in output


def test_main_locate_reports_errors(monkeypatch, capsys):
    """Locate failures print quizzer: ... and exit 1."""
    def boom(*args, **kwargs):
        raise LookupError("No TODO comments")

    monkeypatch.setattr(cli, "locate_module", boom)
    assert cli.main(["locate", "9.9"]) == 1
    assert "No TODO comments" in capsys.readouterr().err


def test_main_quiz_and_run(monkeypatch, capsys, tmp_path: Path):
    """quiz and run forward --questions and can write JSON / --save-locate."""
    monkeypatch.setattr(cli, "locate_module", lambda *args, **kwargs: FAKE_RESULT)
    captured: list[dict] = []

    def fake_generate(located, **kwargs):
        captured.append(kwargs)
        return FAKE_QUIZ

    monkeypatch.setattr(cli, "generate_quiz", fake_generate)
    loc_path = tmp_path / "loc.json"
    loc_path.write_text(json.dumps(FAKE_RESULT.to_dict()), encoding="utf-8")
    quiz_out = tmp_path / "quiz.json"
    assert cli.main(["quiz", str(loc_path), "--format", "json", "-o", str(quiz_out)]) == 0
    assert json.loads(quiz_out.read_text(encoding="utf-8"))["questions"][0]["answer"]
    assert captured[0]["question_count"] == DEFAULT_QUESTION_COUNT

    saved = tmp_path / "saved-loc.json"
    assert cli.main(["run", "1.1", "--questions", "3", "--save-locate", str(saved)]) == 0
    assert captured[1]["question_count"] == 3
    assert json.loads(saved.read_text(encoding="utf-8"))["module"] == "1.1"
    assert "Why this shape?" in capsys.readouterr().out


def test_main_quiz_missing_file(tmp_path: Path, capsys):
    """quiz on a missing locate file exits with a quizzer: error."""
    missing = tmp_path / "nope.json"
    assert cli.main(["quiz", str(missing)]) == 1
    assert capsys.readouterr().err.startswith("quizzer:")


def test_main_testing_quiz(monkeypatch, capsys, tmp_path: Path):
    """testing writes the rule-based public-tests quiz and needs no language model."""
    payload = {
        "module": "tests",
        "model": "rule-based",
        "questions": [{"format": "matching", "question": "Q", "answer": "A"}],
    }
    monkeypatch.setattr(cli, "generate_testing_quiz", lambda *args, **kwargs: payload)
    out = tmp_path / "tests.json"
    assert cli.main(["testing", "--format", "json", "-o", str(out)]) == 0
    assert json.loads(out.read_text(encoding="utf-8"))["module"] == "tests"
    assert "matching" in capsys.readouterr().out


def test_package_exports():
    """The package re-exports the same functions the CLI imports."""
    import quizzer

    assert quizzer.locate_module is cli.locate_module
    assert quizzer.generate_quiz is cli.generate_quiz
    assert quizzer.generate_testing_quiz is cli.generate_testing_quiz


def test_main_locate_against_mini_tree(tmp_path: Path, capsys):
    """End-to-end locate --no-lm-filter on a tiny tree returns the finished snippet."""
    original = write_tree(
        tmp_path / "original",
        {
            "ASSIGNMENT.md": "### 1\n\n> **TODO(1.2)**\n> Loop.\n",
            "src/mod.py": "def run():\n    # TODO(1.2): loop\n    raise NotImplementedError\n",
        },
    )
    finished = write_tree(
        tmp_path / "finished",
        {"src/mod.py": "def run():\n    while True:\n        break\n"},
    )
    code = cli.main(
        [
            "locate",
            "1.2",
            "--finished",
            str(finished),
            "--original",
            str(original),
            "--no-lm-filter",
            "--format",
            "json",
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["snippets"][0]["symbol"] == "run"
    assert "while True" in payload["snippets"][0]["after"]["code"]
