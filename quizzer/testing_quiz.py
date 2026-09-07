"""Rule-based quiz: which assignment module each public test targets."""

from __future__ import annotations

import ast
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from quizzer.locate import (
    DEFAULT_FINISHED_DIR,
    DEFAULT_ORIGINAL_DIR,
    HEADING_RE,
    TODO_RE,
    _qualname_map,
    _read_text,
    canonical_todo_ids,
    collect_todo_hits,
    resolve_assignment_md,
    target_node,
)

NONE_MODULE = "none"
PART_RE = re.compile(r"^##\s+Part\s+(\d+)\b", re.IGNORECASE)
SUBHEAD_RE = re.compile(r"^###\s+(\d+)\.\s+(.*)")
STOPWORDS = {
    "about",
    "after",
    "agent",
    "also",
    "and",
    "are",
    "does",
    "for",
    "from",
    "into",
    "its",
    "not",
    "that",
    "the",
    "this",
    "with",
    "without",
    "when",
    "which",
}


@dataclass
class PublicTest:
    """One ``test_*`` function from the assignment ``tests/`` tree."""

    file: str
    name: str
    description: str
    module: str


def quiz_module_id(todo_id: str) -> str:
    """Drop letter suffixes so ``1.1.a`` becomes ``1.1``."""
    parts = [part for part in todo_id.split(".") if part]
    if parts and parts[-1].isalpha() and len(parts[-1]) == 1:
        parts = parts[:-1]
    if len(parts) >= 2:
        return ".".join(parts[:2])
    return ".".join(parts) if parts else todo_id


def _first_sentence(text: str) -> str:
    collapsed = " ".join(text.split()).strip()
    if not collapsed:
        return ""
    for index, char in enumerate(collapsed):
        if char == "." and (index + 1 == len(collapsed) or collapsed[index + 1] == " "):
            return collapsed[: index + 1]
    return collapsed if collapsed.endswith(".") else collapsed + "."


def _from_name(name: str) -> str:
    stem = name[5:] if name.startswith("test_") else name
    words = stem.replace("_", " ").strip()
    if not words:
        return "Untitled test."
    return words[0].upper() + words[1:] + "."


def _description(node: ast.AST) -> str:
    doc = ast.get_docstring(node) or ""
    return _first_sentence(doc) or _from_name(getattr(node, "name", "test"))


def _names_used(node: ast.AST) -> set[str]:
    found: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name):
            found.add(child.id)
        elif isinstance(child, ast.Attribute):
            found.add(child.attr)
        elif isinstance(child, ast.Constant) and isinstance(child.value, str):
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", child.value):
                found.add(child.value)
    return found


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z][a-z0-9_]{2,}", text.lower())
        if token not in STOPWORDS
    }


def collect_public_tests(tests_dir: Path) -> list[PublicTest]:
    """List module-level ``test_*`` functions under ``tests_dir``."""
    if not tests_dir.is_dir():
        raise FileNotFoundError(f"Tests directory does not exist: {tests_dir}")
    items: list[PublicTest] = []
    for path in sorted(tests_dir.glob("test_*.py")):
        source = _read_text(path)
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            raise ValueError(f"Could not parse {path}: {exc}") from exc
        rel = path.name
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if not node.name.startswith("test_"):
                continue
            items.append(
                PublicTest(
                    file=rel,
                    name=node.name,
                    description=_description(node),
                    module=NONE_MODULE,
                )
            )
    if not items:
        raise LookupError(f"No test_* functions in {tests_dir}")
    return items


GENERIC_NAMES = {
    "Agent",
    "ChessAgent",
    "CodeAgent",
    "make_agent",
    "pytest",
    "run",
}
INFRA_FILES = {
    "test_env.py",
    "test_smoke.py",
    "test_infrastructure.py",
    "test_chess_sandbox.py",
}


def _todo_symbol_modules(original_dir: Path) -> dict[str, set[str]]:
    """Map a symbol tail (``build_prompt``) or TODO-comment word to quiz module ids."""
    mapping: dict[str, set[str]] = defaultdict(set)
    hits: list[dict[str, Any]] = []
    for part in ("1", "2", "3", "4", "5"):
        hits.extend(collect_todo_hits(original_dir, part))
    by_file: dict[Path, list[dict[str, Any]]] = defaultdict(list)
    for hit in hits:
        by_file[hit["path"]].append(hit)
    for path, file_hits in by_file.items():
        source = _read_text(path)
        lines = source.splitlines(keepends=True)
        tree = ast.parse(source)
        names = _qualname_map(tree)
        for hit in file_hits:
            node = target_node(tree, hit["line"], lines)
            modules = {quiz_module_id(todo_id) for todo_id in hit["todo_ids"]}
            keys: set[str] = set()
            if node is not None:
                symbol = names.get(node) or getattr(node, "name", "")
                for part in symbol.split("."):
                    if not part:
                        continue
                    keys.add(part)
                    lower = part.lower()
                    if lower.endswith("_tool"):
                        keys.add(lower[: -len("_tool")])
            for key in keys:
                mapping[key].update(modules)
    return mapping


def _assignment_sections(
    assignment_md: Path | None,
) -> list[tuple[str, str]]:
    """``(module_id, section text)`` from Part / numbered headings and TODOs."""
    if assignment_md is None or not assignment_md.is_file():
        return []
    lines = _read_text(assignment_md).splitlines()
    sections: list[tuple[str, str]] = []
    part = ""
    current_id = ""
    current: list[str] = []

    def flush() -> None:
        nonlocal current_id, current
        if current_id and current:
            sections.append((current_id, "\n".join(current)))
        current = []

    for line in lines:
        part_match = PART_RE.match(line)
        if part_match:
            flush()
            part = part_match.group(1)
            current_id = ""
            current = [line]
            continue
        sub = SUBHEAD_RE.match(line)
        if sub and part:
            flush()
            current_id = f"{part}.{sub.group(1)}"
            current = [line]
            continue
        if HEADING_RE.match(line) and not line.startswith("####"):
            flush()
            current_id = ""
            current = []
            continue
        if current_id:
            current.append(line)
            for match in TODO_RE.finditer(line):
                for todo_id in canonical_todo_ids(match.group("body")):
                    numbered = quiz_module_id(todo_id)
                    if numbered and numbered[0].isdigit():
                        current_id = numbered
    flush()
    return sections


def _prefer_part(path_name: str, module: str) -> int:
    if path_name.startswith("test_chess") and module.startswith("3"):
        return 2
    if path_name.startswith("test_agent") and module[:1] in {"1", "2"}:
        return 2
    return 0


def infer_module(
    node: ast.AST,
    file_name: str,
    symbol_modules: dict[str, set[str]],
    sections: list[tuple[str, str]],
) -> str:
    """Pick the assignment module this test most clearly exercises."""
    if file_name in INFRA_FILES:
        return NONE_MODULE

    scores: dict[str, int] = defaultdict(int)
    name = getattr(node, "name", "")
    doc = ast.get_docstring(node) or ""
    blob = f"{name} {doc}"
    used = _names_used(node)

    for match in TODO_RE.finditer(blob):
        for todo_id in canonical_todo_ids(match.group("body")):
            scores[quiz_module_id(todo_id)] += 12

    for ident in used:
        weight = 1 if ident in GENERIC_NAMES else 4
        for module in symbol_modules.get(ident, ()):
            scores[module] += weight + _prefer_part(file_name, module)

    lowered = blob.lower()
    if "compact" in lowered:
        scores["2.1"] += 8 if "summar" in lowered else 5
        scores["2.2"] += 5
    if "truncat" in lowered:
        scores["1.3"] += 8
    if file_name.startswith("test_agent") and (
        "malformed" in lowered
        or "not valid json" in lowered
        or "unknown_tool" in lowered
        or "unknown tool" in lowered
        or "execute" in used
    ):
        scores["1.3"] += 6
    if "skill" in lowered:
        scores["1.4"] += 8
    if "step_limit" in lowered or "step limit" in lowered:
        scores["1.2"] += 5
    if "play_move" in lowered or "PLAY_MOVE_TOOL" in used:
        scores["3.1"] += 6
    if "format_chess_state" in used or "include_legal_moves" in used or "board_only" in name:
        scores["3.2"] += 8
    if "prompt" in lowered and "skill" not in lowered and "truncat" not in lowered:
        if file_name.startswith("test_chess"):
            scores["3.1"] += 4
        else:
            scores["1.1"] += 4
    if file_name.startswith("test_chess") and (
        "malformed" in lowered or "rejected" in lowered
    ):
        scores["3.1"] += 6

    test_tokens = _tokens(blob)
    for module, text in sections:
        if not _prefer_part(file_name, module):
            continue
        overlap = test_tokens & _tokens(text)
        if overlap:
            scores[module] += min(4, len(overlap))

    if not scores:
        return NONE_MODULE
    return max(scores, key=lambda module: (scores[module], -len(module), module))


def _attach_modules(
    tests_dir: Path,
    original_dir: Path,
    assignment_md: Path | None,
) -> list[PublicTest]:
    symbol_modules = _todo_symbol_modules(original_dir)
    sections = _assignment_sections(assignment_md)
    items: list[PublicTest] = []
    for path in sorted(tests_dir.glob("test_*.py")):
        source = _read_text(path)
        tree = ast.parse(source)
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if not node.name.startswith("test_"):
                continue
            items.append(
                PublicTest(
                    file=path.name,
                    name=node.name,
                    description=_description(node),
                    module=infer_module(node, path.name, symbol_modules, sections),
                )
            )
    if not items:
        raise LookupError(f"No test_* functions in {tests_dir}")
    return items


def _label(index: int) -> str:
    return chr(ord("a") + index)


def build_testing_question(items: list[PublicTest]) -> dict[str, str]:
    """One matching item: each test description mapped to a module id."""
    lines = [
        "Each public test is described in one sentence. Which assignment "
        "module does it target? Write none if the test is setup or "
        "infrastructure rather than a student TODO.",
        "",
    ]
    answer_lines: list[str] = []
    last_file = ""
    for index, item in enumerate(items):
        if item.file != last_file:
            lines.append(f"From `{item.file}`:")
            last_file = item.file
        tag = _label(index)
        lines.append(f"({tag}) {item.description}")
        answer_lines.append(
            f"({tag}) -> {item.module}. {item.file}::{item.name}"
        )
    lines.extend(
        [
            "",
            "Modules are ids such as 1.1, 1.2, 2.1, 3.1. Use none when the "
            "test does not cover a student TODO.",
        ]
    )
    return {
        "format": "matching",
        "dimension": "coupling",
        "question": "\n".join(lines),
        "answer": "\n".join(answer_lines),
        "explanation": (
            "Public tests are the contract for each TODO. Matching a test's "
            "behavior to a module shows whether you know what each part is "
            "supposed to change."
        ),
    }


def generate_testing_quiz(
    finished_dir: str | Path = DEFAULT_FINISHED_DIR,
    original_dir: str | Path = DEFAULT_ORIGINAL_DIR,
    tests_dir: str | Path | None = None,
    assignment_md: str | Path | None = None,
) -> dict[str, Any]:
    """Build the rule-based public-tests quiz from the assignment trees."""
    finished = Path(finished_dir).resolve()
    original = Path(original_dir).resolve()
    tests = Path(tests_dir).resolve() if tests_dir else finished / "tests"
    if not original.is_dir():
        raise FileNotFoundError(f"Original directory does not exist: {original}")
    spec_path = resolve_assignment_md(assignment_md, original, finished)
    items = _attach_modules(tests, original, spec_path)
    question = build_testing_question(items)
    return {
        "module": "tests",
        "model": "rule-based",
        "question_count": 1,
        "questions": [question],
        "tests": [
            {
                "file": item.file,
                "name": item.name,
                "description": item.description,
                "module": item.module,
            }
            for item in items
        ],
    }
