from __future__ import annotations

"""Tests for tools/data_sufficiency.py's check_data_sufficiency - deterministic,
no network/LLM. Scoped to .csv only; .dat files are a mixed domain-specific
format in this project (confirmed against real AQT_electrolyte data - some are
numeric matrices, others are "key = value" metadata pairs) and are never
touched by this check.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from manuscript_system.domain.models import SourceAsset  # noqa: E402
from manuscript_system.persistence.repositories import new_id  # noqa: E402
from manuscript_system.tools.data_sufficiency import check_data_sufficiency  # noqa: E402


def make_asset(project_id: str, relative_path: str) -> SourceAsset:
    return SourceAsset(
        artifact_id=new_id("ARTIFACT"), project_id=project_id, relative_path=relative_path,
        checksum_sha256="deadbeef", size_bytes=100, media_type="text/csv",
    )


def test_clean_csv_produces_no_findings(tmp_path):
    project_id = "PROJECT-TEST"
    (tmp_path / "data.csv").write_text("x,y\n1.0,2.0\n3.0,4.0\n")
    assets = [make_asset(project_id, "data.csv")]
    findings = check_data_sufficiency(assets, tmp_path, project_id)
    assert findings == []


def test_empty_csv_is_flagged(tmp_path):
    project_id = "PROJECT-TEST"
    (tmp_path / "empty.csv").write_text("x,y\n")
    assets = [make_asset(project_id, "empty.csv")]
    findings = check_data_sufficiency(assets, tmp_path, project_id)
    assert len(findings) == 1
    assert findings[0].rule_id == "data_file_empty"
    assert findings[0].severity == "HIGH"


def test_non_finite_values_are_flagged(tmp_path):
    project_id = "PROJECT-TEST"
    (tmp_path / "bad.csv").write_text("x,y\n1.0,inf\n2.0,3.0\n")
    assets = [make_asset(project_id, "bad.csv")]
    findings = check_data_sufficiency(assets, tmp_path, project_id)
    assert len(findings) == 1
    assert findings[0].rule_id == "data_file_non_finite_values"


def test_unreadable_csv_is_flagged(tmp_path):
    project_id = "PROJECT-TEST"
    # inconsistent field counts per row - pandas' default C parser raises
    (tmp_path / "malformed.csv").write_text("a,b,c\n1,2\n3,4,5,6,7\n")
    assets = [make_asset(project_id, "malformed.csv")]
    findings = check_data_sufficiency(assets, tmp_path, project_id)
    assert len(findings) == 1
    assert findings[0].rule_id == "data_file_unreadable"


def test_missing_file_is_flagged_unreadable(tmp_path):
    project_id = "PROJECT-TEST"
    assets = [make_asset(project_id, "does_not_exist.csv")]
    findings = check_data_sufficiency(assets, tmp_path, project_id)
    assert len(findings) == 1
    assert findings[0].rule_id == "data_file_unreadable"


def test_dat_files_are_never_touched(tmp_path):
    """.dat files are a mixed domain-specific format here (some are numeric
    matrices, some are key=value metadata pairs) - a generic tabular parser
    can't validate both without misclassifying one as malformed, so this
    check is scoped to .csv only."""
    project_id = "PROJECT-TEST"
    (tmp_path / "metadata.dat").write_text("HF_energy(Ha)     = -937.6178973550\n")
    assets = [SourceAsset(
        artifact_id=new_id("ARTIFACT"), project_id=project_id, relative_path="metadata.dat",
        checksum_sha256="deadbeef", size_bytes=100, media_type="text/plain",
    )]
    findings = check_data_sufficiency(assets, tmp_path, project_id)
    assert findings == []


def test_mixed_assets_only_checks_csv(tmp_path):
    project_id = "PROJECT-TEST"
    (tmp_path / "good.csv").write_text("x,y\n1.0,2.0\n")
    (tmp_path / "notes.txt").write_text("just some notes")
    assets = [
        make_asset(project_id, "good.csv"),
        SourceAsset(
            artifact_id=new_id("ARTIFACT"), project_id=project_id, relative_path="notes.txt",
            checksum_sha256="deadbeef", size_bytes=100, media_type="text/plain",
        ),
    ]
    findings = check_data_sufficiency(assets, tmp_path, project_id)
    assert findings == []
