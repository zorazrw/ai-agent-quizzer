"""Tests for quizzer.locate helpers and locate_module."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from quizzer.locate import (
    LocateResult,
    Snippet,
    Span,
    _innermost_containing,
    _leading_comment_start,
    _qualname_map,
    _trailing_comment_end,
    apply_snippet_filter,
    build_filter_user_message,
    canonical_todo_ids,
    collect_todo_hits,
    expand_todo_body,
    extract_spec,
    find_symbol,
    format_markdown,
    iter_python_files,
    locate_module,
    module_matches,
    narrow_to_changes,
    parse_filter_json,
    parse_module_query,
    resolve_assignment_md,
    span_for_node,
    target_node,
)
from quizzer.prompts import FILTER_SYSTEM_PROMPT
from quizzer.tests.helpers import write_tree

SAMPLE = '''\
class Agent:
    def __init__(self):
        self.ready = True

        # TODO(1.1.a): store history

    def build_prompt(self):
        # TODO(1.1.a): messages
        raise NotImplementedError

    def nested(self):
        def inner():
            return 1
        return inner()


# TODO(3.1.a): schema
PLAY_MOVE_TOOL: dict = {}
'''


def _tree(src: str = SAMPLE) -> ast.AST:
    return ast.parse(src)


def _lines(src: str = SAMPLE) -> list[str]:
    return src.splitlines(keepends=True)


def _todo_line(needle: str, src: str = SAMPLE) -> int:
    return next(
        index
        for index, line in enumerate(src.splitlines(), start=1)
        if needle in line
    )


def test_parse_module_query_normalizes_wrappers():
    assert parse_module_query(" 1.1.A ") == "1.1.a"
    assert parse_module_query("TODO(1.2)") == "1.2"
    assert parse_module_query("Part 3.1") == "3.1"
    assert parse_module_query("TODO(Part 1.1)") == "1.1"


def test_parse_module_query_rejects_garbage():
    with pytest.raises(ValueError, match="Not a module index"):
        parse_module_query("build_prompt")


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ("", [""]),
        ("1.1.a", ["1.1.a"]),
        ("Part 3.1.b", ["3.1.b"]),
        ("3.3-4", ["3.3", "3.4"]),
        ("3.4-3", ["3.3", "3.4"]),
        ("1.1, 1.2", ["1.1", "1.2"]),
    ],
)
def test_expand_todo_body(body, expected):
    assert expand_todo_body(body) == expected


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ("", ["3.4"]),
        ("2", ["2.1"]),
        ("3", ["3.1"]),
        ("3, 3", ["3.1"]),
    ],
)
def test_canonical_todo_ids_applies_aliases(body, expected):
    assert canonical_todo_ids(body) == expected


@pytest.mark.parametrize(
    ("todo_id", "query", "matched"),
    [
        ("1.1.a", "1.1", True),
        ("1.1.a", "1.1.a", True),
        ("1.1.b", "1.1.a", False),
        ("1.10", "1.1", False),
        ("3.1", "3", True),
        ("1.2", "1.1", False),
    ],
)
def test_module_matches(todo_id, query, matched):
    assert module_matches(todo_id, query) is matched


def test_leading_comment_start_walks_contiguous_hashes():
    lines = ["x = 1\n", "# a\n", "# b\n", "y = 2\n"]
    assert _leading_comment_start(lines, 4) == 2
    assert _leading_comment_start(lines, 1) == 1


def test_trailing_comment_end_skips_blanks_and_stops_at_code():
    lines = [
        "        self.ready = True\n",
        "\n",
        "        # TODO leftover\n",
        "\n",
        "    def next(self):\n",
    ]
    assert _trailing_comment_end(lines, 1) == 3


def test_iter_python_files_skips_cache_dirs(tmp_path: Path):
    write_tree(
        tmp_path,
        {
            "keep.py": "x = 1\n",
            "pkg/mod.py": "y = 2\n",
            "__pycache__/ignored.py": "z = 3\n",
            ".venv/lib/nope.py": "z = 4\n",
            ".mypy_cache/ignored.py": "z = 5\n",
        },
    )
    found = {path.name for path in iter_python_files(tmp_path)}
    assert found == {"keep.py", "mod.py"}


def test_qualname_map_names_methods_and_assignments():
    names = set(_qualname_map(_tree()).values())
    assert "Agent" in names
    assert "Agent.__init__" in names
    assert "Agent.build_prompt" in names
    assert "Agent.nested.inner" in names
    assert "PLAY_MOVE_TOOL" in names
    assign_src = "FOO = {}\n"
    assert "FOO" in _qualname_map(_tree(assign_src)).values()


def test_innermost_containing_prefers_nested_function():
    src = SAMPLE
    tree = _tree(src)
    inner_line = _todo_line("return 1")
    node = _innermost_containing(tree, inner_line, (ast.FunctionDef,))
    assert node is not None
    assert node.name == "inner"


def test_target_node_binds_trailing_todo_to_previous_method():
    todo_line = _todo_line("TODO(1.1.a): store history")
    node = target_node(_tree(), todo_line, _lines())
    assert isinstance(node, ast.FunctionDef)
    assert node.name == "__init__"


def test_target_node_binds_todo_inside_function():
    todo_line = _todo_line("TODO(1.1.a): messages")
    node = target_node(_tree(), todo_line, _lines())
    assert isinstance(node, ast.FunctionDef)
    assert node.name == "build_prompt"


def test_target_node_binds_todo_above_assignment():
    todo_line = _todo_line("TODO(3.1.a)")
    node = target_node(_tree(), todo_line, _lines())
    assert isinstance(node, ast.AnnAssign)


def test_target_node_binds_todo_above_function():
    src = (
        "class Agent:\n"
        "    # TODO(1.2): implement\n"
        "    def run(self):\n"
        "        raise NotImplementedError\n"
    )
    todo_line = _todo_line("TODO(1.2)", src)
    node = target_node(_tree(src), todo_line, _lines(src))
    assert isinstance(node, ast.FunctionDef)
    assert node.name == "run"


def test_find_symbol_exact_and_unambiguous_tail():
    tree = _tree()
    assert find_symbol(tree, "Agent.build_prompt").name == "build_prompt"
    assert find_symbol(tree, "PLAY_MOVE_TOOL") is not None
    assert find_symbol(tree, "missing") is None
    assert find_symbol(tree, "build_prompt").name == "build_prompt"


def test_find_symbol_refuses_ambiguous_tails():
    src = (
        "class A:\n"
        "    def foo(self):\n"
        "        return 1\n"
        "\n"
        "class B:\n"
        "    def foo(self):\n"
        "        return 2\n"
    )
    tree = ast.parse(src)
    assert find_symbol(tree, "A.foo").name == "foo"
    assert find_symbol(tree, "foo") is None


def test_span_for_node_includes_leading_comments_on_assignments():
    lines = _lines()
    tree = _tree()
    node = find_symbol(tree, "PLAY_MOVE_TOOL")
    with_comments = span_for_node(lines, node, include_comments=True)
    without = span_for_node(lines, node, include_comments=False)
    assert "TODO(3.1.a)" in with_comments.code
    assert with_comments.code.strip().endswith("PLAY_MOVE_TOOL: dict = {}")
    assert "TODO(3.1.a)" not in without.code


def test_span_for_node_includes_trailing_comments_on_functions():
    lines = _lines()
    tree = _tree()
    node = find_symbol(tree, "Agent.__init__")
    span = span_for_node(lines, node, include_comments=True)
    assert "TODO(1.1.a): store history" in span.code


def test_narrow_to_changes_keeps_short_spans():
    before = Span(1, 3, "a\nb\nc\n")
    after = Span(1, 4, "a\nb\nX\nc\n")
    assert narrow_to_changes(before, after) == (before, after)


def test_narrow_to_changes_trims_long_identical_prefix():
    prefix = [f"line-{i}\n" for i in range(40)]
    before = Span(1, 42, "".join(prefix + ["OLD\n", "tail\n"]))
    after = Span(1, 42, "".join(prefix + ["NEW\n", "tail\n"]))
    narrowed_before, narrowed_after = narrow_to_changes(before, after)
    assert "OLD" in narrowed_before.code
    assert "NEW" in narrowed_after.code
    assert narrowed_before.code.count("line-") < 40


def test_narrow_to_changes_identical_long_spans_unchanged():
    body = "".join(f"line-{i}\n" for i in range(45))
    span = Span(1, 45, body)
    assert narrow_to_changes(span, span) == (span, span)


def test_collect_todo_hits_filters_by_query(tmp_path: Path):
    write_tree(tmp_path, {"mod.py": SAMPLE})
    hits_11 = collect_todo_hits(tmp_path, "1.1")
    comments = {hit["comment"] for hit in hits_11}
    assert any("store history" in comment for comment in comments)
    assert any("messages" in comment for comment in comments)
    hits_31 = collect_todo_hits(tmp_path, "3.1.a")
    assert len(hits_31) == 1
    assert "schema" in hits_31[0]["comment"]


def test_collect_todo_hits_keeps_following_comments_and_part_alias(tmp_path: Path):
    write_tree(
        tmp_path,
        {
            "mod.py": (
                "# TODO(Part 2): compact the context\n"
                "# keep extra explanation\n"
                "def maybe_compact():\n"
                "    pass\n"
            )
        },
    )
    hits = collect_todo_hits(tmp_path, "2.1")
    assert len(hits) == 1
    assert "compact the context" in hits[0]["comment"]
    assert "keep extra explanation" in hits[0]["comment"]


def test_extract_spec_keeps_matching_heading_section(tmp_path: Path):
    path = tmp_path / "ASSIGNMENT.md"
    path.write_text(
        "### 1. Prompt\n\n> **TODO(1.1.a)**\n> Build it.\n\n"
        "### 2. Loop\n\n> **TODO(1.2)**\n> Run it.\n",
        encoding="utf-8",
    )
    spec = extract_spec(path, "1.1")
    assert "TODO(1.1.a)" in spec
    assert "TODO(1.2)" not in spec
    assert extract_spec(tmp_path / "missing.md", "1.1") == ""


def test_locate_module_mini_assignment(mini_assignment: tuple[Path, Path]):
    finished, original = mini_assignment
    result = locate_module("1.1", finished, original)
    symbols = {snippet.symbol for snippet in result.snippets}
    assert symbols == {"Agent.__init__", "Agent.build_prompt"}
    assert "TODO(1.1.a)" in result.spec
    prompt = next(s for s in result.snippets if s.symbol == "Agent.build_prompt")
    assert "raise NotImplementedError" in prompt.before.code
    assert '"role": "system"' in prompt.after.code
    history = next(s for s in result.snippets if s.symbol == "Agent.__init__")
    assert "self.history = []" in history.after.code

    only_a = locate_module("1.1.a", finished, original)
    assert only_a.module == "1.1.a"
    assert {s.symbol for s in only_a.snippets} == symbols


def test_locate_module_schema_assignment(mini_assignment: tuple[Path, Path]):
    finished, original = mini_assignment
    schema = locate_module("3.1.a", finished, original)
    assert schema.snippets[0].symbol == "PLAY_MOVE_TOOL"
    assert "play_move" in schema.snippets[0].after.code


def test_locate_module_merges_todos_on_same_symbol(tmp_path: Path):
    original = write_tree(
        tmp_path / "original",
        {
            "mod.py": (
                "def foo():\n"
                "    # TODO(1.1.a): first\n"
                "    x = 1\n"
                "    # TODO(1.1.b): second\n"
                "    y = 2\n"
            )
        },
    )
    finished = write_tree(
        tmp_path / "finished",
        {"mod.py": "def foo():\n    x = 1\n    y = 2\n"},
    )
    result = locate_module("1.1", finished, original)
    assert len(result.snippets) == 1
    snippet = result.snippets[0]
    assert snippet.symbol == "foo"
    assert snippet.todo_ids == ["1.1.a", "1.1.b"]
    assert any("first" in comment for comment in snippet.todo_comments)
    assert any("second" in comment for comment in snippet.todo_comments)


def test_locate_module_errors(tmp_path: Path, mini_assignment: tuple[Path, Path]):
    finished, original = mini_assignment
    with pytest.raises(FileNotFoundError, match="Original directory"):
        locate_module("1.1", finished, tmp_path / "nope")
    with pytest.raises(FileNotFoundError, match="Finished directory"):
        locate_module("1.1", tmp_path / "nope", original)
    with pytest.raises(LookupError, match="No TODO comments"):
        locate_module("9.9", finished, original)

    missing_finished = write_tree(tmp_path / "empty_finished", {"ASSIGNMENT.md": ""})
    with pytest.raises(FileNotFoundError, match="Finished copy missing"):
        locate_module("1.1", missing_finished, original)


def test_locate_module_missing_symbol(tmp_path: Path):
    original = write_tree(
        tmp_path / "original",
        {
            "src/mod.py": (
                "def foo():\n    # TODO(1.2): implement\n    raise NotImplementedError\n"
            )
        },
    )
    finished = write_tree(tmp_path / "finished", {"src/mod.py": "def bar():\n    return 1\n"})
    with pytest.raises(LookupError, match="was not found"):
        locate_module("1.2", finished, original)


def test_locate_module_unbound_todo(tmp_path: Path):
    original = write_tree(
        tmp_path / "original",
        {"mod.py": "# TODO(1.1): orphan comment with no nearby symbol\n"},
    )
    finished = write_tree(
        tmp_path / "finished",
        {"mod.py": "# leftover\n"},
    )
    with pytest.raises(LookupError, match="Could not bind TODO"):
        locate_module("1.1", finished, original)


def test_locate_module_spec_falls_back_to_finished(tmp_path: Path):
    original = write_tree(
        tmp_path / "original",
        {"mod.py": "def foo():\n    # TODO(1.2): implement\n    pass\n"},
    )
    finished = write_tree(
        tmp_path / "finished",
        {
            "ASSIGNMENT.md": "### Loop\n\n> **TODO(1.2)**\n> Do the loop.\n",
            "mod.py": "def foo():\n    return 1\n",
        },
    )
    result = locate_module("1.2", finished, original)
    assert "Do the loop" in result.spec


def test_locate_result_to_dict_and_format_markdown(mini_assignment: tuple[Path, Path]):
    finished, original = mini_assignment
    result = locate_module("1.2", finished, original)
    payload = result.to_dict()
    json.dumps(payload)
    assert payload["module"] == "1.2"
    assert payload["snippets"][0]["symbol"] == "Agent.run"
    assert payload["assignment_md"].endswith("ASSIGNMENT.md")
    markdown = format_markdown(result)
    assert "# Module 1.2" in markdown
    assert "Assignment:" in markdown
    assert "## Spec" in markdown
    assert "**Before**" in markdown
    assert "**After**" in markdown
    assert "Agent.run" in markdown


def test_format_markdown_omits_spec_when_empty():
    result = LocateResult(
        module="9.9",
        finished_dir="/f",
        original_dir="/o",
        spec="",
        snippets=[
            Snippet(
                todo_ids=["9.9"],
                todo_comments=["# TODO(9.9): x"],
                file="a.py",
                symbol="foo",
                before=Span(1, 1, "def foo():\n    pass\n"),
                after=Span(1, 2, "def foo():\n    return 1\n"),
            )
        ],
    )
    markdown = format_markdown(result)
    assert "## Spec" not in markdown
    assert "```py" in markdown


def test_locate_real_assignment_1_1(repo_root: Path):
    original = repo_root / "assignment1" / "original"
    finished = repo_root / "assignment1" / "finished"
    if not original.is_dir() or not finished.is_dir():
        pytest.skip("exemplar assignment1 trees are not present")
    result = locate_module("1.1", finished, original)
    symbols = {snippet.symbol for snippet in result.snippets}
    assert "Agent.build_prompt" in symbols
    assert "CodeAgent.__init__" in symbols
    prompt = next(s for s in result.snippets if s.symbol == "Agent.build_prompt")
    assert "raise NotImplementedError" in prompt.before.code
    assert "self.system_prompt" in prompt.after.code


def test_resolve_assignment_md_prefers_explicit_path(tmp_path: Path):
    original = write_tree(tmp_path / "original", {"ASSIGNMENT.md": "orig\n"})
    finished = write_tree(tmp_path / "finished", {"ASSIGNMENT.md": "fin\n"})
    custom = tmp_path / "custom.md"
    custom.write_text("# custom\n", encoding="utf-8")
    resolved = resolve_assignment_md(custom, original, finished)
    assert resolved == custom.resolve()
    assert resolve_assignment_md(None, original, finished) == (original / "ASSIGNMENT.md").resolve()
    with pytest.raises(FileNotFoundError, match="Assignment spec"):
        resolve_assignment_md(tmp_path / "missing.md", original, finished)


def test_locate_module_uses_explicit_assignment_md(tmp_path: Path):
    original = write_tree(
        tmp_path / "original",
        {
            "ASSIGNMENT.md": "### Wrong\n\n> **TODO(1.2)**\n> Ignore me.\n",
            "mod.py": "def foo():\n    # TODO(1.2): implement\n    pass\n",
        },
    )
    finished = write_tree(
        tmp_path / "finished",
        {"mod.py": "def foo():\n    return 1\n"},
    )
    assignment = tmp_path / "spec.md"
    assignment.write_text("### Loop\n\n> **TODO(1.2)**\n> Do the loop.\n", encoding="utf-8")
    result = locate_module("1.2", finished, original, assignment_md=assignment)
    assert "Do the loop" in result.spec
    assert "Ignore me" not in result.spec
    assert result.assignment_md == str(assignment.resolve())


def test_parse_filter_json_and_apply_trim():
    snippets = [
        Snippet(
            todo_ids=["1.1.a"],
            todo_comments=["# TODO(1.1.a)"],
            file="a.py",
            symbol="Agent.build_prompt",
            before=Span(1, 2, "def build_prompt(self):\n    raise NotImplementedError\n"),
            after=Span(
                1,
                6,
                "def build_prompt(self):\n"
                "    memory = self.context_summary\n"
                "    return [system, user, *history]\n",
            ),
        ),
        Snippet(
            todo_ids=["1.1.a"],
            todo_comments=["# TODO(1.1.a)"],
            file="a.py",
            symbol="unrelated",
            before=Span(1, 1, "x = 1\n"),
            after=Span(1, 1, "x = 2\n"),
        ),
    ]
    selected = parse_filter_json(
        json.dumps(
            {
                "selected": [
                    {
                        "index": 1,
                        "before_code": "def build_prompt(self):\n    raise NotImplementedError",
                        "after_code": (
                            "def build_prompt(self):\n"
                            "    return [system, user, *history]\n"
                        ),
                    }
                ]
            }
        ),
        snippet_count=2,
    )
    kept = apply_snippet_filter(snippets, selected)
    assert len(kept) == 1
    assert kept[0].symbol == "Agent.build_prompt"
    assert "context_summary" not in kept[0].after.code
    assert kept[0].after.code.endswith("\n")


def test_parse_filter_json_rejects_bad_index():
    with pytest.raises(ValueError, match="invalid snippet index"):
        parse_filter_json('{"selected": [{"index": 9}]}', snippet_count=1)
    with pytest.raises(ValueError, match="`selected` list"):
        parse_filter_json('{"selected": "nope"}', snippet_count=1)


def test_build_filter_user_message_includes_spec():
    result = LocateResult(
        module="1.1",
        finished_dir="/f",
        original_dir="/o",
        spec="Build the prompt, not compaction.",
        assignment_md="/ASSIGNMENT.md",
        snippets=[
            Snippet(
                todo_ids=["1.1.a"],
                todo_comments=["# TODO(1.1.a): messages"],
                file="src/agent.py",
                symbol="Agent.build_prompt",
                before=Span(1, 1, "raise NotImplementedError\n"),
                after=Span(1, 2, "return messages\n"),
            )
        ],
    )
    text = build_filter_user_message(result)
    assert "Module: 1.1" in text
    assert "Build the prompt, not compaction." in text
    assert "snippet 1: Agent.build_prompt" in text
    assert "context compaction" in FILTER_SYSTEM_PROMPT


def test_locate_module_lm_filter_drops_and_trims(tmp_path: Path):
    original = write_tree(
        tmp_path / "original",
        {
            "ASSIGNMENT.md": (
                "### Prompt\n\n> **TODO(1.1.a)**\n> Build history and the prompt.\n"
            ),
            "mod.py": (
                "def build_prompt():\n"
                "    # TODO(1.1.a): messages\n"
                "    raise NotImplementedError\n"
                "\n"
                "def compact_context():\n"
                "    # TODO(1.1.a): leftover label on the wrong helper\n"
                "    raise NotImplementedError\n"
            ),
        },
    )
    finished = write_tree(
        tmp_path / "finished",
        {
            "mod.py": (
                "def build_prompt():\n"
                "    memory = context_summary\n"
                "    return [system, user, *history]\n"
                "\n"
                "def compact_context():\n"
                "    return summary\n"
            )
        },
    )
    payload = {
        "selected": [
            {
                "index": 1,
                "before_code": (
                    "def build_prompt():\n"
                    "    raise NotImplementedError\n"
                ),
                "after_code": (
                    "def build_prompt():\n"
                    "    return [system, user, *history]\n"
                ),
            }
        ]
    }

    class FakeClient:
        def __init__(self):
            self.messages = self

        def create(self, **kwargs):
            self.kwargs = kwargs
            return SimpleNamespace(
                content=[SimpleNamespace(text=json.dumps(payload))]
            )

    client = FakeClient()
    result = locate_module(
        "1.1.a",
        finished,
        original,
        filter_with_lm=True,
        client=client,
    )
    assert [snippet.symbol for snippet in result.snippets] == ["build_prompt"]
    assert "context_summary" not in result.snippets[0].after.code
    assert "*history]" in result.snippets[0].after.code
    assert client.kwargs["system"] == FILTER_SYSTEM_PROMPT
    assert "Build history and the prompt" in client.kwargs["messages"][0]["content"]


def test_locate_module_lm_filter_empty_selection(tmp_path: Path):
    original = write_tree(
        tmp_path / "original",
        {"mod.py": "def foo():\n    # TODO(1.2): x\n    pass\n"},
    )
    finished = write_tree(
        tmp_path / "finished",
        {"mod.py": "def foo():\n    return 1\n"},
    )

    class FakeClient:
        def __init__(self):
            self.messages = self

        def create(self, **kwargs):
            return SimpleNamespace(
                content=[SimpleNamespace(text='{"selected": []}')]
            )

    with pytest.raises(LookupError, match="kept no snippets"):
        locate_module(
            "1.2",
            finished,
            original,
            filter_with_lm=True,
            client=FakeClient(),
        )
