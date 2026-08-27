from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

DEFAULT_EXCLUDES = frozenset({
    ".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ipynb_checkpoints",
    "node_modules", ".venv", "venv", ".DS_Store",
})


class BoundaryError(ValueError):
    """Raised when a project path fails the read-only intake boundary check."""


@dataclass(frozen=True)
class ScannedFile:
    relative_path: str
    absolute_path: Path
    checksum_sha256: str
    size_bytes: int
    media_type: str


def validate_project_path(raw_path: str) -> Path:
    if not raw_path or not raw_path.startswith("/"):
        raise BoundaryError("An absolute project path is required.")
    path = Path(raw_path).resolve()
    if not path.exists():
        raise BoundaryError(f"Path does not exist: {path}")
    if not path.is_dir():
        raise BoundaryError(f"Path is not a directory: {path}")
    return path


def _checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


_MEDIA_TYPES = {
    ".py": "text/x-python", ".ipynb": "application/x-ipynb+json",
    ".csv": "text/csv", ".dat": "text/plain", ".txt": "text/plain",
    ".md": "text/markdown", ".yaml": "application/yaml", ".yml": "application/yaml",
    ".json": "application/json", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".png": "image/png", ".pdf": "application/pdf",
}


def _media_type(path: Path) -> str:
    return _MEDIA_TYPES.get(path.suffix.lower(), "application/octet-stream")


def scan_project(root: Path, *, excludes: frozenset[str] = DEFAULT_EXCLUDES) -> list[ScannedFile]:
    """Read-only recursive scan. Never writes to `root`.

    Symlinks that resolve outside `root` are skipped rather than followed, so
    the scan cannot be tricked into reading (or, if a writer existed, writing)
    outside the declared project boundary.
    """
    results: list[ScannedFile] = []
    resolved_root = root.resolve()
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if any(part in excludes for part in path.relative_to(root).parts):
            continue
        try:
            path.resolve().relative_to(resolved_root)
        except ValueError:
            continue
        results.append(ScannedFile(
            relative_path=str(path.relative_to(root)),
            absolute_path=path,
            checksum_sha256=_checksum(path),
            size_bytes=path.stat().st_size,
            media_type=_media_type(path),
        ))
    return results
