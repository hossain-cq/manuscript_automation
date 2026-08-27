from __future__ import annotations

"""Tests for assessment.py's discover_project - a domain-agnostic structure
summary built from already-persisted SourceAsset rows, no network/LLM needed.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from manuscript_system.domain.models import SourceAsset  # noqa: E402
from manuscript_system.graphs.assessment import discover_project  # noqa: E402
from manuscript_system.persistence.database import connect  # noqa: E402
from manuscript_system.persistence.repositories import Repository, new_id  # noqa: E402


def make_repo(tmp_path: Path) -> Repository:
    return Repository(connect(str(tmp_path / "test.sqlite")))


def make_context(project_id: str, run_id: str) -> dict:
    return {
        "project_id": project_id, "run_id": run_id, "thread_id": f"thread-{run_id}",
        "workflow_name": "project_assessment", "status": "RUNNING", "current_stage": "DISCOVERY",
        "created_at": "2026-01-01T00:00:00Z", "updated_at": "2026-01-01T00:00:00Z",
    }


def make_asset(project_id: str, relative_path: str, size_bytes: int = 100) -> SourceAsset:
    return SourceAsset(
        artifact_id=new_id("ARTIFACT"), project_id=project_id, relative_path=relative_path,
        checksum_sha256="deadbeef", size_bytes=size_bytes, media_type="application/octet-stream",
    )


def test_disorganized_project_trips_all_four_findings(tmp_path):
    repo = make_repo(tmp_path)
    project_id = "PROJECT-TEST"
    # only documents, no README, no environment file, no code, no data/figures
    repo.add_source_assets([
        make_asset(project_id, "notes.txt"),
        make_asset(project_id, "summary.md"),
    ])
    state = {"context": make_context(project_id, "RUN-TEST")}
    result = discover_project(state, repo)

    findings = repo.get_findings(project_id)
    rule_ids = {f.rule_id for f in findings}
    assert rule_ids == {
        "missing_readme", "missing_environment_file", "no_code_files_found", "no_data_or_figures_found",
    }
    assert all(f.severity == "MEDIUM" for f in findings)
    assert all(not f.blocking for f in findings)

    knowledge_map = repo.get_knowledge_map(project_id)
    assert knowledge_map.map_id == result["knowledge_map_id"]
    assert knowledge_map.has_readme is False
    assert knowledge_map.has_environment_file is False
    assert knowledge_map.total_assets == 2


def test_well_organized_project_trips_zero_findings(tmp_path):
    """Mirrors the real AQT_electrolyte project structure - a README,
    environment.yml, and code/data/figures spread across subdirectories."""
    repo = make_repo(tmp_path)
    project_id = "PROJECT-TEST"
    repo.add_source_assets([
        make_asset(project_id, "README.md"),
        make_asset(project_id, "environment.yml"),
        make_asset(project_id, "src/main.py"),
        make_asset(project_id, "notebooks/analysis.ipynb"),
        make_asset(project_id, "data/results.csv"),
        make_asset(project_id, "results/plot.png"),
    ])
    state = {"context": make_context(project_id, "RUN-TEST")}
    result = discover_project(state, repo)

    findings = repo.get_findings(project_id)
    assert findings == []
    assert result["finding_ids"] == []

    knowledge_map = repo.get_knowledge_map(project_id)
    assert knowledge_map.has_readme is True
    assert knowledge_map.has_environment_file is True
    assert knowledge_map.asset_counts_by_role == {
        "DOCUMENT": 1, "CONFIG": 1, "CODE": 2, "DATA": 1, "FIGURE": 1,
    }
    assert knowledge_map.total_assets == 6
