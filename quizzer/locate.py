"""Step 1: find a TODO module and extract original vs finished snippets."""

from __future__ import annotations

import ast
import difflib
import re
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable

from quizzer.lm import DEFAULT_MAX_TOKENS, DEFAULT_MODEL, anthropic_client, parse_json_object, response_text
from quizzer.prompts import FILTER_SYSTEM_PROMPT


TODO_RE = re.compile(
    r"TODO\(\s*(?:Part\s+)?(?P<body>[^)]*?)\s*\)",
    re.IGNORECASE,
)
HEADING_RE = re.compile(r"^(#{1,6})\s+")
SKIP_DIR_NAMES = {".git", ".venv", "__pycache__", ".mypy_cache", ".pytest_cache"}
DEFAULT_FINISHED_DIR = "assignment1/finished"
DEFAULT_ORIGINAL_DIR = "assignment1/original"

# Bare or coarse labels in the scaffold that correspond to a real module id.
TODO_ALIASES = {
    "": ["3.4"],  # `# TODO()` above RUN_PYTHON_TOOL
    "2": ["2.1"],
    "3": ["3.1"],
}


@dataclass
class Span:
    """Inclusive line range plus the source text for that range."""

    start_line: int
    end_line: int
    code: str


@dataclass
class Snippet:
    """One original-vs-finished symbol, plus the TODO comments that pointed at it."""

    todo_ids: list[str]
    todo_comments: list[str]
    file: str
    symbol: str
    before: Span
    after: Span


@dataclass
class LocateResult:
    """Everything step 1 returns for one assignment module."""

    module: str
    finished_dir: str
    original_dir: str
    spec: str
    assignment_md: str = ""
    snippets: list[Snippet] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Flatten this result (and nested Span dataclasses) into JSON-safe dicts."""
        return {
            "module": self.module,
            "finished_dir": self.finished_dir,
            "original_dir": self.original_dir,
            "spec": self.spec,
            "assignment_md": self.assignment_md,
            "snippets": [
                {
                    "todo_ids": snippet.todo_ids,
                    "todo_comments": snippet.todo_comments,
                    "file": snippet.file,
                    "symbol": snippet.symbol,
                    "before": asdict(snippet.before),
                    "after": asdict(snippet.after),
                }
                for snippet in self.snippets
            ],
        }


def parse_module_query(raw: str) -> str:
    """Normalize ``TODO(1.1)`` / ``Part 3.1`` / ``1.1.A`` into a canonical id."""
    module = raw.strip().lower()
    if module.startswith("todo"):
        module = module.split("(", 1)[-1].rstrip(")").strip()
    module = re.sub(r"^part\s+", "", module)
    if not re.fullmatch(r"\d+(?:\.\d+)*(?:\.[a-z])?", module):
        raise ValueError(f"Not a module index: {raw!r}")
    return module


def expand_todo_body(body: str) -> list[str]:
    """Turn ``3.3-4`` / ``Part 3.1.a`` / comma lists into ids such as ``3.3``, ``3.4``."""

    text = body.strip()
    if not text:
        return [""]

    text = re.sub(r"^Part\s+", "", text, flags=re.IGNORECASE)
    ids: list[str] = []
    ranged = re.fullmatch(r"(\d+(?:\.\d+)*)\.(\d+)-(\d+)", text)
    if ranged:
        prefix, start, end = ranged.group(1), int(ranged.group(2)), int(ranged.group(3))
        lo, hi = (start, end) if start <= end else (end, start)
        return [f"{prefix}.{n}" for n in range(lo, hi + 1)]

    for piece in re.split(r"\s*,\s*", text):
        piece = piece.strip().lower()
        if piece:
            ids.append(piece)
    return ids or [""]


def canonical_todo_ids(body: str) -> list[str]:
    """Expand a TODO body, then map aliases (``3`` → ``3.1``, empty → ``3.4``)."""
    ids: list[str] = []
    for item in expand_todo_body(body):
        ids.extend(TODO_ALIASES.get(item, [item]))
    return list(dict.fromkeys(ids))


def module_matches(todo_id: str, query: str) -> bool:
    """Prefix match: ``1.1`` includes ``1.1.a``; ``1.1.a`` matches only that sub-item."""

    return todo_id == query or todo_id.startswith(query + ".")


def _read_text(path: Path) -> str:
    """Read a file as UTF-8 text."""
    return path.read_text(encoding="utf-8")


def _slice_lines(lines: list[str], start: int, end: int) -> str:
    """Join 1-indexed inclusive ``[start, end]`` lines and force a trailing newline."""
    return "".join(lines[start - 1 : end]).rstrip() + "\n"


def _leading_comment_start(lines: list[str], start_line: int) -> int:
    """Walk upward from ``start_line`` so contiguous ``#`` comments are included."""

    index = start_line - 1
    while index > 0 and lines[index - 1].strip().startswith("#"):
        index -= 1
    return index + 1


def iter_python_files(root: Path) -> Iterable[Path]:
    """Yield ``*.py`` paths under ``root``, skipping venv/cache directories."""
    for path in sorted(root.rglob("*.py")):
        if any(part in SKIP_DIR_NAMES for part in path.parts):
            continue
        yield path


def _qualname_map(tree: ast.AST) -> dict[ast.AST, str]:
    """Map class/function/assignment nodes to dotted names like ``Agent.build_prompt``."""
    names: dict[ast.AST, str] = {}

    def walk(node: ast.AST, prefix: str) -> None:
        """Recurse, prefixing nested defs/assigns with the enclosing class or function."""
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                qualified = f"{prefix}.{child.name}" if prefix else child.name
                names[child] = qualified
                walk(child, qualified)
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qualified = f"{prefix}.{child.name}" if prefix else child.name
                names[child] = qualified
                walk(child, qualified)
            elif isinstance(child, (ast.Assign, ast.AnnAssign)):
                target = child.targets[0] if isinstance(child, ast.Assign) else child.target
                if isinstance(target, ast.Name):
                    qualified = f"{prefix}.{target.id}" if prefix else target.id
                    names[child] = qualified
                walk(child, prefix)
            else:
                walk(child, prefix)

    walk(tree, "")
    return names


def _innermost_containing(
    tree: ast.AST, line: int, kinds: tuple[type[ast.AST], ...]
) -> ast.AST | None:
    """Among ``kinds`` whose span covers ``line``, pick the smallest (innermost) node."""
    best: ast.AST | None = None
    best_span = None
    for node in ast.walk(tree):
        if not isinstance(node, kinds):
            continue
        start = getattr(node, "lineno", None)
        end = getattr(node, "end_lineno", None)
        if start is None or end is None:
            continue
        if start <= line <= end:
            span = end - start
            if best is None or span < best_span:
                best = node
                best_span = span
    return best


def _indent(line: str) -> int:
    """Count leading spaces (used to tell trailing comments from the next ``def``)."""
    return len(line) - len(line.lstrip(" "))


def _previous_function(tree: ast.AST, line: int) -> ast.AST | None:
    """The function whose ``end_lineno`` is closest to, but still before, ``line``."""
    best: ast.AST | None = None
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        end = getattr(node, "end_lineno", None)
        if end is not None and end < line:
            if best is None or end > best.end_lineno:
                best = node
    return best


def _next_declaration(tree: ast.AST, line: int) -> ast.AST | None:
    """Earliest def/assign at or after ``line`` — the stub a leading TODO documents."""

    candidates: list[ast.AST] = []
    for node in ast.walk(tree):
        if not isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Assign, ast.AnnAssign)
        ):
            continue
        start = getattr(node, "lineno", None)
        if start is not None and start >= line:
            candidates.append(node)
    if not candidates:
        return None
    return min(candidates, key=lambda node: node.lineno)


def target_node(tree: ast.AST, todo_line: int, lines: list[str]) -> ast.AST | None:
    """Bind a TODO line to its function: inside it, the next stub, or a trailing note."""
    inside = _innermost_containing(
        tree,
        todo_line,
        (ast.FunctionDef, ast.AsyncFunctionDef),
    )
    if inside is not None:
        return inside

    nxt = _next_declaration(tree, todo_line)
    prev = _previous_function(tree, todo_line)
    # Comments indented deeper than the next `def` are trailing notes in the
    # previous method: AST end_lineno stops at the last statement, so a TODO
    # after that statement is not "inside" the function.
    if prev is not None and nxt is not None:
        todo_indent = _indent(lines[todo_line - 1])
        next_indent = _indent(lines[nxt.lineno - 1])
        if todo_indent > next_indent:
            return prev
    return nxt or prev


def _trailing_comment_end(lines: list[str], end_line: int) -> int:
    """Extend past blank lines to include ``#`` comments after the last statement."""

    last = end_line
    for index in range(end_line, len(lines)):
        stripped = lines[index].strip()
        if stripped == "":
            continue
        if stripped.startswith("#"):
            last = index + 1
            continue
        break
    return last


def find_symbol(tree: ast.AST, symbol: str) -> ast.AST | None:
    """Look up ``symbol`` in the finished tree; fall back to a unique unqualified tail."""
    names = _qualname_map(tree)
    for node, qualified in names.items():
        if qualified == symbol:
            return node
    # Fall back to the unqualified tail, e.g. `__init__` vs `Agent.__init__`.
    tail = symbol.rsplit(".", 1)[-1]
    matches = [
        node
        for node, qualified in names.items()
        if qualified == tail or qualified.endswith("." + tail)
    ]
    if len(matches) == 1:
        return matches[0]
    return None


def span_for_node(lines: list[str], node: ast.AST, include_comments: bool) -> Span:
    """Line range and source for ``node``; optionally glue on leading/trailing ``#`` comments."""
    start = node.lineno
    end = node.end_lineno or node.lineno
    if include_comments:
        start = _leading_comment_start(lines, start)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            end = _trailing_comment_end(lines, end)
    return Span(start_line=start, end_line=end, code=_slice_lines(lines, start, end))


LONG_SYMBOL_LINES = 40
DIFF_CONTEXT_LINES = 4


def narrow_to_changes(before: Span, after: Span) -> tuple[Span, Span]:
    """If a symbol is long, keep only the differing region plus a few context lines."""

    before_count = before.end_line - before.start_line + 1
    after_count = after.end_line - after.start_line + 1
    if max(before_count, after_count) <= LONG_SYMBOL_LINES:
        return before, after

    before_lines = before.code.splitlines(keepends=True)
    after_lines = after.code.splitlines(keepends=True)
    matcher = difflib.SequenceMatcher(a=before_lines, b=after_lines, autojunk=False)
    before_lo = before_hi = after_lo = after_hi = None
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        before_lo = i1 if before_lo is None else min(before_lo, i1)
        before_hi = i2 if before_hi is None else max(before_hi, i2)
        after_lo = j1 if after_lo is None else min(after_lo, j1)
        after_hi = j2 if after_hi is None else max(after_hi, j2)
    if before_lo is None or after_lo is None:
        return before, after

    before_lo = max(0, before_lo - DIFF_CONTEXT_LINES)
    after_lo = max(0, after_lo - DIFF_CONTEXT_LINES)
    before_hi = min(len(before_lines), before_hi + DIFF_CONTEXT_LINES)
    after_hi = min(len(after_lines), after_hi + DIFF_CONTEXT_LINES)
    return (
        Span(
            start_line=before.start_line + before_lo,
            end_line=before.start_line + before_hi - 1,
            code="".join(before_lines[before_lo:before_hi]),
        ),
        Span(
            start_line=after.start_line + after_lo,
            end_line=after.start_line + after_hi - 1,
            code="".join(after_lines[after_lo:after_hi]),
        ),
    )


def collect_todo_hits(original_dir: Path, query: str) -> list[dict[str, Any]]:
    """Scan original ``*.py`` files for ``TODO(...)`` comments that match ``query``."""
    hits: list[dict[str, Any]] = []
    for path in iter_python_files(original_dir):
        text = _read_text(path)
        for line_no, line in enumerate(text.splitlines(), start=1):
            match = TODO_RE.search(line)
            if not match:
                continue
            todo_ids = [
                todo_id
                for todo_id in canonical_todo_ids(match.group("body"))
                if module_matches(todo_id, query)
            ]
            if not todo_ids:
                continue
            comment_lines = [line.rstrip()]
            all_lines = text.splitlines()
            follow = line_no
            while follow < len(all_lines):
                nxt = all_lines[follow].rstrip()
                if nxt.strip().startswith("#") and not TODO_RE.search(nxt):
                    comment_lines.append(nxt)
                    follow += 1
                    continue
                break
            hits.append(
                {
                    "path": path,
                    "line": line_no,
                    "todo_ids": todo_ids,
                    "comment": "\n".join(comment_lines),
                }
            )
    return hits


def extract_spec(assignment_md: Path, query: str) -> str:
    """Pull ASSIGNMENT.md heading sections whose body mentions this module's TODO ids."""
    if not assignment_md.is_file():
        return ""
    lines = _read_text(assignment_md).splitlines(keepends=True)
    matching_sections: list[str] = []
    current: list[str] = []
    current_matches = False

    def flush() -> None:
        """If the current heading section matched the query, keep it; then reset."""
        nonlocal current, current_matches
        if current_matches and current:
            matching_sections.append("".join(current).rstrip())
        current = []
        current_matches = False

    for line in lines:
        if HEADING_RE.match(line) and current:
            flush()
        current.append(line)
        for match in TODO_RE.finditer(line):
            if any(
                module_matches(todo_id, query)
                for todo_id in canonical_todo_ids(match.group("body"))
            ):
                current_matches = True
    flush()
    return "\n\n".join(matching_sections).strip()


def resolve_assignment_md(
    assignment_md: str | Path | None,
    original: Path,
    finished: Path,
) -> Path | None:
    """Use ``assignment_md`` if given, else ASSIGNMENT.md under original then finished."""

    if assignment_md is not None:
        path = Path(assignment_md).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Assignment spec does not exist: {path}")
        return path
    for candidate in (original / "ASSIGNMENT.md", finished / "ASSIGNMENT.md"):
        if candidate.is_file():
            return candidate
    return None


def _ensure_newline(code: str) -> str:
    return code if code.endswith("\n") else code + "\n"


def build_filter_user_message(result: LocateResult) -> str:
    """Prompt the LM with this module's spec and the rule-based snippets."""

    parts = [
        f"Module: {result.module}",
        "",
        "Assignment specification for this module (from ASSIGNMENT.md):",
        result.spec or "(no ASSIGNMENT.md section found)",
        "",
        "Snippets below were selected by matching TODO comments. Keep only "
        "what implements THIS module. Strip neighboring-module code that "
        "shares the same function (for example Part 2 compaction inside "
        "build_prompt).",
        "",
    ]
    for index, snippet in enumerate(result.snippets, start=1):
        ids = ", ".join(snippet.todo_ids)
        parts.extend(
            [
                f"--- snippet {index}: {snippet.symbol} "
                f"({snippet.file}; TODO {ids}) ---",
                "Original TODO comment:",
                "\n\n".join(snippet.todo_comments),
                "",
                "BEFORE",
                snippet.before.code,
                "",
                "AFTER",
                snippet.after.code,
                "",
            ]
        )
    return "\n".join(parts).strip()


def parse_filter_json(text: str, snippet_count: int) -> list[dict[str, Any]]:
    """Parse the LM's selected-snippet list (1-based indices)."""

    payload = parse_json_object(text)
    selected = payload.get("selected")
    if not isinstance(selected, list):
        raise ValueError("Filter JSON must contain a `selected` list.")
    seen: set[int] = set()
    items: list[dict[str, Any]] = []
    for entry in selected:
        if not isinstance(entry, dict):
            raise ValueError("Each filter `selected` item must be an object.")
        index = entry.get("index")
        if not isinstance(index, int) or index < 1 or index > snippet_count:
            raise ValueError(f"Filter returned invalid snippet index: {index!r}")
        if index in seen:
            continue
        seen.add(index)
        items.append(entry)
    return items


def apply_snippet_filter(
    snippets: list[Snippet], selected: list[dict[str, Any]]
) -> list[Snippet]:
    """Keep selected snippets and replace their code when the LM trimmed it."""

    kept: list[Snippet] = []
    for entry in selected:
        snippet = snippets[entry["index"] - 1]
        before = snippet.before
        after = snippet.after
        before_code = entry.get("before_code")
        after_code = entry.get("after_code")
        if isinstance(before_code, str) and before_code.strip():
            before = Span(
                start_line=before.start_line,
                end_line=before.end_line,
                code=_ensure_newline(before_code),
            )
        if isinstance(after_code, str) and after_code.strip():
            after = Span(
                start_line=after.start_line,
                end_line=after.end_line,
                code=_ensure_newline(after_code),
            )
        kept.append(replace(snippet, before=before, after=after))
    return kept


def filter_snippets_with_lm(
    result: LocateResult,
    *,
    model: str = DEFAULT_MODEL,
    client: Any | None = None,
) -> LocateResult:
    """Drop or trim snippets that do not match this module's ASSIGNMENT.md spec."""

    api = client or anthropic_client()
    try:
        response = api.messages.create(
            model=model,
            max_tokens=DEFAULT_MAX_TOKENS,
            system=FILTER_SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": build_filter_user_message(result)},
            ],
        )
    except Exception as exc:
        raise RuntimeError(
            f"Language model request failed ({type(exc).__name__}): {exc}"
        ) from exc
    selected = parse_filter_json(response_text(response), len(result.snippets))
    snippets = apply_snippet_filter(result.snippets, selected)
    if not snippets:
        raise LookupError(
            f"LM filter kept no snippets for module {result.module!r}."
        )
    return replace(result, snippets=snippets)


def locate_module(
    module: str,
    finished_dir: str | Path = DEFAULT_FINISHED_DIR,
    original_dir: str | Path = DEFAULT_ORIGINAL_DIR,
    assignment_md: str | Path | None = None,
    *,
    filter_with_lm: bool = False,
    model: str = DEFAULT_MODEL,
    client: Any | None = None,
) -> LocateResult:
    """Main entry: spec + original-vs-finished snippets for one assignment module."""
    query = parse_module_query(module)
    finished = Path(finished_dir).resolve()
    original = Path(original_dir).resolve()
    if not original.is_dir():
        raise FileNotFoundError(f"Original directory does not exist: {original}")
    if not finished.is_dir():
        raise FileNotFoundError(f"Finished directory does not exist: {finished}")

    spec_path = resolve_assignment_md(assignment_md, original, finished)
    spec = extract_spec(spec_path, query) if spec_path is not None else ""

    hits = collect_todo_hits(original, query)
    if not hits:
        raise LookupError(
            f"No TODO comments in {original} match module {query!r}."
        )

    grouped: dict[tuple[str, str], Snippet] = {}
    for hit in hits:
        rel = hit["path"].relative_to(original).as_posix()
        finished_path = finished / rel
        if not finished_path.is_file():
            raise FileNotFoundError(
                f"Finished copy missing for {rel}: expected {finished_path}"
            )

        original_src = _read_text(hit["path"])
        finished_src = _read_text(finished_path)
        original_lines = original_src.splitlines(keepends=True)
        finished_lines = finished_src.splitlines(keepends=True)
        original_tree = ast.parse(original_src)
        finished_tree = ast.parse(finished_src)
        names = _qualname_map(original_tree)

        node = target_node(original_tree, hit["line"], original_lines)
        if node is None:
            raise LookupError(f"Could not bind TODO on {rel}:{hit['line']} to a symbol")
        symbol = names.get(node) or getattr(node, "name", rel)
        before = span_for_node(original_lines, node, include_comments=True)

        finished_node = find_symbol(finished_tree, symbol)
        if finished_node is None:
            raise LookupError(
                f"Symbol {symbol!r} from {rel} was not found in the finished file"
            )
        after = span_for_node(finished_lines, finished_node, include_comments=False)
        before, after = narrow_to_changes(before, after)

        key = (rel, symbol)
        if key in grouped:
            existing = grouped[key]
            for todo_id in hit["todo_ids"]:
                if todo_id not in existing.todo_ids:
                    existing.todo_ids.append(todo_id)
            existing.todo_comments.append(hit["comment"])
        else:
            grouped[key] = Snippet(
                todo_ids=list(hit["todo_ids"]),
                todo_comments=[hit["comment"]],
                file=rel,
                symbol=symbol,
                before=before,
                after=after,
            )

    snippets = list(grouped.values())
    snippets.sort(key=lambda item: (item.file, item.before.start_line))
    result = LocateResult(
        module=query,
        finished_dir=str(finished),
        original_dir=str(original),
        spec=spec,
        assignment_md=str(spec_path) if spec_path is not None else "",
        snippets=snippets,
    )
    if filter_with_lm:
        result = filter_snippets_with_lm(result, model=model, client=client)
    return result


def format_markdown(result: LocateResult) -> str:
    """Render spec plus each snippet's TODO comments, before code, and after code."""
    parts = [
        f"# Module {result.module}",
        "",
        f"Original: `{result.original_dir}`",
        f"Finished: `{result.finished_dir}`",
    ]
    if result.assignment_md:
        parts.append(f"Assignment: `{result.assignment_md}`")
    parts.append("")
    if result.spec:
        parts.extend(["## Spec", "", result.spec, ""])
    parts.append("## Snippets")
    for snippet in result.snippets:
        ids = ", ".join(snippet.todo_ids)
        parts.extend(
            [
                "",
                f"### `{snippet.symbol}` in `{snippet.file}` ({ids})",
                "",
            ]
        )
        for comment in snippet.todo_comments:
            parts.extend(["```", comment, "```", ""])
        parts.extend(
            [
                f"**Before** `{snippet.file}:{snippet.before.start_line}"
                f"-{snippet.before.end_line}`",
                "",
                f"```{snippet.file.rsplit('.', 1)[-1]}",
                snippet.before.code.rstrip(),
                "```",
                "",
                f"**After** `{snippet.file}:{snippet.after.start_line}"
                f"-{snippet.after.end_line}`",
                "",
                f"```{snippet.file.rsplit('.', 1)[-1]}",
                snippet.after.code.rstrip(),
                "```",
            ]
        )
    return "\n".join(parts).rstrip() + "\n"
