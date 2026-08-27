from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from ..settings import get_settings

"""Loads a named style from configs/citation_styles.yaml - the formatting
counterpart to tools/journals.py's load_journal_profile. A journal profile's
citation.style_id points here; manuscript_evaluation.py uses the loaded
style's validation block to check bibliography-entry formatting (title/DOI
presence) against what the target journal actually requires."""


def load_citation_style(style_id: str | None) -> dict[str, Any] | None:
    if not style_id:
        return None
    path = Path(get_settings().configs_path) / "citation_styles.yaml"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data.get("styles", {}).get(style_id)
