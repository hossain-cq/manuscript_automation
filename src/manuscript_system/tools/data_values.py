from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

"""Deterministic extraction of numeric values from a raw research-data
project, for cross-checking against numbers stated in a manuscript.

Scans .csv (pandas), .dat (whitespace-delimited, # comments), and .ipynb
(cell source + executed output text). .ipynb specifically because checking
the real AQT_electrolyte + wiley/ pair by hand showed the manuscript's
reported energies (e.g. "Exact (CASCI) = -945.0931 Ha") live in notebook
*output* text, not in any standalone .csv/.dat file - grepped directly and
confirmed absent from data/raw/, present in notebooks/LiPF6_new.ipynb's
executed cell output.
"""

MIN_SIGNIFICANT_DIGITS = 4
NUMBER_RE = re.compile(r"[-+]?\d+\.\d{2,}")

_EXCLUDED_DIR_PARTS = {".git", "__pycache__", ".ipynb_checkpoints"}


@dataclass(frozen=True)
class ExtractedValue:
    value: float
    source_relative_path: str
    context: str


def significant_digits(text: str) -> int:
    """Count digits ignoring sign/decimal-point and leading zeros - "1.50"
    (3) is far less distinctive than "-945.0931" (7). Used on both sides
    (manuscript numbers and data-file numbers) so low-precision, common
    values like bond lengths don't produce false-positive matches purely by
    coincidence."""
    digits = re.sub(r"[^0-9]", "", text).lstrip("0")
    return len(digits) or 1


def _is_distinctive(text: str) -> bool:
    return significant_digits(text) >= MIN_SIGNIFICANT_DIGITS


def find_distinctive_numbers(text: str) -> list[str]:
    return [token for token in NUMBER_RE.findall(text) if _is_distinctive(token)]


def _extract_from_csv(path: Path, root: Path) -> list[ExtractedValue]:
    values: list[ExtractedValue] = []
    try:
        df = pd.read_csv(path)
    except Exception:
        return values
    rel = str(path.relative_to(root))
    for column in df.columns:
        if not pd.api.types.is_numeric_dtype(df[column]):
            continue
        for raw_value in df[column].dropna():
            text = repr(float(raw_value))
            if _is_distinctive(text):
                values.append(ExtractedValue(value=float(raw_value), source_relative_path=rel, context=f"column {column!r}"))
    return values


def _extract_from_dat(path: Path, root: Path) -> list[ExtractedValue]:
    values: list[ExtractedValue] = []
    rel = str(path.relative_to(root))
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return values
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        for token in find_distinctive_numbers(stripped):
            values.append(ExtractedValue(value=float(token), source_relative_path=rel, context=stripped[:80]))
    return values


def _output_text(output: dict) -> str:
    if "text" in output:
        data = output["text"]
        return "".join(data) if isinstance(data, list) else str(data)
    text_plain = output.get("data", {}).get("text/plain")
    if text_plain is not None:
        return "".join(text_plain) if isinstance(text_plain, list) else str(text_plain)
    return ""


def _extract_from_ipynb(path: Path, root: Path) -> list[ExtractedValue]:
    values: list[ExtractedValue] = []
    rel = str(path.relative_to(root))
    try:
        notebook = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except (OSError, json.JSONDecodeError):
        return values
    for cell in notebook.get("cells", []):
        for output in cell.get("outputs", []):
            text = _output_text(output)
            for token in find_distinctive_numbers(text):
                values.append(ExtractedValue(value=float(token), source_relative_path=rel, context=text.strip()[:80]))
    return values


_EXTRACTORS = {".csv": _extract_from_csv, ".dat": _extract_from_dat, ".ipynb": _extract_from_ipynb}


def extract_numeric_values(root: str | Path) -> list[ExtractedValue]:
    root = Path(root)
    values: list[ExtractedValue] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        extractor = _EXTRACTORS.get(path.suffix.lower())
        if extractor is None:
            continue
        if any(part in _EXCLUDED_DIR_PARTS for part in path.relative_to(root).parts):
            continue
        values.extend(extractor(path, root))
    return values
