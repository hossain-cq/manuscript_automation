from __future__ import annotations

import hashlib
from pathlib import Path


class ArtifactStore:
    """Content-addressed filesystem store for artifacts the system generates.

    Raw project files are never copied here - tools/filesystem.py only records
    their checksums. This store is for derived artifacts (reports, figures,
    manuscript blocks) produced in later phases.
    """

    def __init__(self, root: str) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def put_bytes(self, data: bytes, *, suffix: str = "") -> str:
        digest = hashlib.sha256(data).hexdigest()
        target = self.root / f"{digest}{suffix}"
        if not target.exists():
            target.write_bytes(data)
        return digest

    def path_for(self, digest: str, *, suffix: str = "") -> Path:
        return self.root / f"{digest}{suffix}"
