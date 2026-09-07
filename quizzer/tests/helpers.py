from __future__ import annotations

from pathlib import Path


def write_tree(root: Path, files: dict[str, str]) -> Path:
    """Create ``root`` with the given relative paths and UTF-8 contents."""
    for relative, contents in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents, encoding="utf-8")
    return root
