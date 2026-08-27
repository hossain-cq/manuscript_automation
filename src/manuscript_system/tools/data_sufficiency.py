from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..domain.models import Finding, SourceAsset
from ..persistence.repositories import new_id

"""Deterministic data-file sanity checks: is a .csv asset actually readable
as tabular data, non-empty, and free of non-finite (NaN/Inf) values.

Reimplements the relevant few lines of literature_and_figures.py's
generic_data_checks directly against real SourceAsset/Finding, rather than
adapting to that module's own parallel DataAsset/Repository types - not
worth a second type system for logic this small.

Scoped to .csv only. .dat files in this domain are a mixed format - some are
genuine whitespace-delimited numeric matrices, others are "key = value"
metadata pairs (confirmed against the real AQT_electrolyte project) - a
generic tabular parser can't validate both without misclassifying one as
malformed. Domain-specific formats like these are evidence_extraction.py's
job (checklist-driven), not a generic sufficiency check.
"""


def check_data_sufficiency(assets: list[SourceAsset], project_root: Path, project_id: str) -> list[Finding]:
    findings: list[Finding] = []
    csv_assets = [a for a in assets if Path(a.relative_path).suffix.lower() == ".csv"]

    for asset in csv_assets:
        path = project_root / asset.relative_path
        try:
            frame = pd.read_csv(path)
        except Exception as exc:
            findings.append(Finding(
                finding_id=new_id("FINDING"), project_id=project_id, severity="HIGH",
                rule_id="data_file_unreadable",
                message=f"Could not parse '{asset.relative_path}' as tabular data: {exc}",
            ))
            continue

        if frame.empty:
            findings.append(Finding(
                finding_id=new_id("FINDING"), project_id=project_id, severity="HIGH",
                rule_id="data_file_empty", message=f"'{asset.relative_path}' has no rows.",
            ))
            continue

        numeric = frame.select_dtypes(include=[np.number]).to_numpy()
        if numeric.size and not np.isfinite(numeric).all():
            findings.append(Finding(
                finding_id=new_id("FINDING"), project_id=project_id, severity="HIGH",
                rule_id="data_file_non_finite_values",
                message=f"'{asset.relative_path}' contains non-finite (NaN/Inf) numeric values.",
            ))

    return findings
