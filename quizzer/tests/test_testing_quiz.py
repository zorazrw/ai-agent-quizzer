"""Tests for the rule-based public-tests quiz."""

from __future__ import annotations

from pathlib import Path

from quizzer.testing_quiz import (
    NONE_MODULE,
    collect_public_tests,
    generate_testing_quiz,
    quiz_module_id,
)
from quizzer.tests.helpers import write_tree


def test_quiz_module_id_drops_letter_suffix():
    """Letter suffixes collapse; two-part ids stay as-is."""
    assert quiz_module_id("1.1.a") == "1.1"
    assert quiz_module_id("3.3.b") == "3.3"
    assert quiz_module_id("1.2") == "1.2"


def test_collect_public_tests_uses_docstring_or_name(tmp_path: Path):
    """A docstring's first sentence is the description; otherwise the function name."""
    tests = write_tree(
        tmp_path / "tests",
        {
            "test_loop.py": (
                "def test_run_stops():\n"
                '    """The agent stops when finished.\\n\\nMore detail."""\n'
                "    pass\n"
                "\n"
                "def helper():\n"
                "    pass\n"
                "\n"
                "def test_no_doc():\n"
                "    pass\n"
            )
        },
    )
    items = collect_public_tests(tests)
    assert [item.name for item in items] == ["test_run_stops", "test_no_doc"]
    assert items[0].description == "The agent stops when finished."
    assert items[1].description == "No doc."


def test_generate_testing_quiz_maps_symbol_to_module(tmp_path: Path):
    """A test that calls a TODO-bound function is labeled with that module."""
    original = write_tree(
        tmp_path / "original",
        {
            "ASSIGNMENT.md": (
                "## Part 1: Agent\n\n"
                "### 1. Build the prompt\n\n"
                "> **TODO(1.1.a)**\n"
                "> History.\n\n"
                "### 2. Run the loop\n\n"
                "> **TODO(1.2)**\n"
                "> ReAct loop.\n"
            ),
            "src/agent.py": (
                "class Agent:\n"
                "    def build_prompt(self):\n"
                "        # TODO(1.1.a): messages\n"
                "        raise NotImplementedError\n"
                "    def run(self):\n"
                "        # TODO(1.2): loop\n"
                "        raise NotImplementedError\n"
            ),
        },
    )
    finished = write_tree(
        tmp_path / "finished",
        {
            "src/agent.py": (
                "class Agent:\n"
                "    def build_prompt(self):\n"
                "        return []\n"
                "    def run(self):\n"
                "        while True:\n"
                "            break\n"
            ),
            "tests/test_agent.py": (
                "def test_prompt_includes_task():\n"
                '    """The prompt still contains the task statement."""\n'
                "    Agent().build_prompt()\n"
                "\n"
                "def test_run_hits_the_limit():\n"
                '    """A run that never finishes raises at the step limit."""\n'
                "    Agent().run()\n"
                "\n"
                "def test_package_importable():\n"
                "    assert True\n"
            ),
        },
    )
    quiz = generate_testing_quiz(finished, original)
    by_name = {row["name"]: row["module"] for row in quiz["tests"]}
    assert by_name["test_prompt_includes_task"] == "1.1"
    assert by_name["test_run_hits_the_limit"] == "1.2"
    assert by_name["test_package_importable"] == NONE_MODULE
    question = quiz["questions"][0]
    assert question["format"] == "matching"
    assert "The prompt still contains the task statement." in question["question"]
    assert "(a) -> 1.1" in question["answer"]
    assert quiz["model"] == "rule-based"
    assert quiz["module"] == "tests"
