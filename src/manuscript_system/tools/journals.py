from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from ..settings import get_settings

"""Shared journal-profile loading, used by both manuscript_evaluation.py
(citation/structure checks against a target journal) and assessment.py
(allocating a manuscript plan's sections from a target journal's required
sections). Moved here from manuscript_evaluation.py once assessment.py
needed the same lookup - a single source rather than two copies."""


def load_journal_profile(journal_id: str | None) -> dict[str, Any] | None:
    if not journal_id:
        return None
    path = Path(get_settings().configs_path) / "journals.yaml"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data.get("journals", {}).get(journal_id)
