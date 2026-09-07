"""Put the quizzer repo root on sys.path so `import quizzer` works."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from quizzer.tests.helpers import write_tree


@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture
def mini_assignment(tmp_path: Path) -> tuple[Path, Path]:
    """A tiny original/finished pair covering trailing TODOs, functions, and schemas."""

    original_src = '''\
class Agent:
    def __init__(self):
        self.ready = True

        # TODO(1.1.a): store the transcript

    def build_prompt(self):
        """Assemble messages."""
        # TODO(1.1.a): return system, user, history
        raise NotImplementedError

    def run(self):
        # TODO(1.2): ReAct loop
        raise NotImplementedError


# TODO(3.1.a): play_move schema
PLAY_MOVE_TOOL: dict = {}
'''
    finished_src = '''\
class Agent:
    def __init__(self):
        self.ready = True

        self.history = []

    def build_prompt(self):
        """Assemble messages."""
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.task_prompt},
            *self.history,
        ]

    def run(self):
        while not self.finished:
            self.history.append(self.query_language_model())


PLAY_MOVE_TOOL: dict = {
    "type": "function",
    "function": {"name": "play_move"},
}
'''
    assignment_md = """\
### 1. Build the prompt

Standing instructions come first.

> **TODO(1.1.a)**
> Maintain history and build the prompt.

### 2. Run the loop

> **TODO(1.2)**
> Run the ReAct loop.

### 3. Chess tools

> **TODO(3.1.a)**
> Define play_move.
"""
    original = write_tree(
        tmp_path / "original",
        {
            "ASSIGNMENT.md": assignment_md,
            "src/agent.py": original_src,
        },
    )
    finished = write_tree(
        tmp_path / "finished",
        {
            "ASSIGNMENT.md": assignment_md,
            "src/agent.py": finished_src,
        },
    )
    return finished, original
