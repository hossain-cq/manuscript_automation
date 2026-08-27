from __future__ import annotations

"""Tests for assessment.py's create_completion_plan_node/create_manuscript_plan_node.

These operate entirely on already-persisted data (Findings, Claims,
ClaimAssessmentRecords) - no network or LLM calls needed, so tests seed the
repository directly rather than running run_audits for real.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from manuscript_system.domain.models import Claim, ClaimAssessmentRecord, Finding  # noqa: E402
from manuscript_system.graphs.assessment import create_completion_plan_node, create_manuscript_plan_node  # noqa: E402
from manuscript_system.persistence.database import connect  # noqa: E402
from manuscript_system.persistence.repositories import Repository, new_id  # noqa: E402


def make_repo(tmp_path: Path) -> Repository:
    return Repository(connect(str(tmp_path / "test.sqlite")))


def make_context(project_id: str, run_id: str) -> dict:
    return {
        "project_id": project_id, "run_id": run_id, "thread_id": f"thread-{run_id}",
        "workflow_name": "project_assessment", "status": "RUNNING", "current_stage": "X",
        "created_at": "2026-01-01T00:00:00Z", "updated_at": "2026-01-01T00:00:00Z",
    }


# --- completion plan ---------------------------------------------------------


def test_completion_plan_includes_missing_checklist_findings(tmp_path):
    repo = make_repo(tmp_path)
    project_id = "PROJECT-TEST"
    repo.add_finding(Finding(
        finding_id=new_id("FINDING"), project_id=project_id, severity="MEDIUM",
        rule_id="missing_checklist_evidence:basis_set_convergence",
        message="No files matched the 'basis_set_convergence' checklist item.",
        blocking=False,
    ))
    state = {"context": make_context(project_id, "RUN-TEST")}
    result = create_completion_plan_node(state, repo)

    tasks = repo.get_completion_tasks(project_id)
    assert len(tasks) == 1
    assert tasks[0].category == "MISSING_DATA"
    assert tasks[0].priority == "REQUIRED"
    assert result["completion_plan_id"]


def test_completion_plan_dedupes_near_identical_assessor_findings(tmp_path):
    """The real AQT_electrolyte run showed all 3 assessors flagging almost
    the same gap in different words for the same claim - this must collapse
    to one task, not three."""
    repo = make_repo(tmp_path)
    project_id = "PROJECT-TEST"
    claim_id = "CLAIM-X"
    repo.add_claims([Claim(
        claim_id=claim_id, project_id=project_id, text="Dissociation curves computed",
        claim_type="RESULT", importance="CENTRAL",
    )])
    phrasings = [
        "No quantitative validation of reported results",
        "Insufficient quantitative validation of reported results",
        "Lack of quantitative validation for the reported results",
    ]
    repo.add_claim_assessments([
        ClaimAssessmentRecord(
            record_id=new_id("CA"), project_id=project_id, run_id="RUN-TEST", assessor_id=f"assessor_{i}",
            claim_id=claim_id, label="SUPPORTED_WITH_LIMITATIONS", score=0.6,
            reasoning="...", missing_evidence=[phrasing],
        )
        for i, phrasing in enumerate(phrasings)
    ])
    state = {"context": make_context(project_id, "RUN-TEST")}
    create_completion_plan_node(state, repo)

    tasks = repo.get_completion_tasks(project_id)
    assert len(tasks) == 1  # 3 near-identical phrasings collapse to 1 task
    assert tasks[0].priority == "REQUIRED"  # claim importance is CENTRAL
    assert tasks[0].affected_claim_ids == [claim_id]


def test_completion_plan_recommended_priority_for_supporting_claim(tmp_path):
    repo = make_repo(tmp_path)
    project_id = "PROJECT-TEST"
    claim_id = "CLAIM-Y"
    repo.add_claims([Claim(
        claim_id=claim_id, project_id=project_id, text="Qubit count trend",
        claim_type="RESULT", importance="SUPPORTING",
    )])
    repo.add_claim_assessments([ClaimAssessmentRecord(
        record_id=new_id("CA"), project_id=project_id, run_id="RUN-TEST", assessor_id="a1",
        claim_id=claim_id, label="SUPPORTED_WITH_LIMITATIONS", score=0.6,
        missing_evidence=["No benchmark comparison"],
    )])
    state = {"context": make_context(project_id, "RUN-TEST")}
    create_completion_plan_node(state, repo)

    tasks = repo.get_completion_tasks(project_id)
    assert tasks[0].priority == "RECOMMENDED"


# --- manuscript plan ----------------------------------------------------------


def test_manuscript_plan_falls_back_to_default_imrad_without_journal(tmp_path):
    repo = make_repo(tmp_path)
    project_id = "PROJECT-TEST"
    repo.add_claims([Claim(
        claim_id="CLAIM-1", project_id=project_id, text="A result", claim_type="RESULT", importance="CENTRAL",
    )])
    state = {"context": make_context(project_id, "RUN-TEST"), "target_journal_id": None}
    result = create_manuscript_plan_node(state, repo)

    sections = repo.get_planned_sections(project_id)
    assert [s.name for s in sections] == ["Introduction", "Methods", "Results", "Discussion", "Conclusion"]
    results_section = next(s for s in sections if s.name == "Results")
    assert results_section.claim_ids == ["CLAIM-1"]
    assert result["manuscript_plan_id"]


def test_manuscript_plan_uses_real_journal_required_sections(tmp_path):
    repo = make_repo(tmp_path)
    project_id = "PROJECT-TEST"
    repo.add_claims([Claim(
        claim_id="CLAIM-1", project_id=project_id, text="A result", claim_type="RESULT", importance="CENTRAL",
    )])
    state = {"context": make_context(project_id, "RUN-TEST"), "target_journal_id": "advanced_quantum_technologies"}
    create_manuscript_plan_node(state, repo)

    sections = repo.get_planned_sections(project_id)
    names = [s.name for s in sections]
    # The real profile combines results+discussion into one section - the
    # claim must still be allocated there via keyword match, not exact equality.
    assert any("results" in n.lower() for n in names)
    combined = next(s for s in sections if "results" in s.name.lower())
    assert "CLAIM-1" in combined.claim_ids
    # front/back matter (title, abstract, references, ...) must be excluded
    assert not any(n.lower() in ("title", "abstract", "references") for n in names)


def test_manuscript_plan_allocates_claims_by_type(tmp_path):
    repo = make_repo(tmp_path)
    project_id = "PROJECT-TEST"
    repo.add_claims([
        Claim(claim_id="CLAIM-RESULT", project_id=project_id, text="x", claim_type="RESULT", importance="CENTRAL"),
        Claim(claim_id="CLAIM-FACT", project_id=project_id, text="x", claim_type="FACT", importance="SUPPORTING"),
        Claim(claim_id="CLAIM-INTERP", project_id=project_id, text="x", claim_type="INTERPRETATION", importance="SUPPORTING"),
    ])
    state = {"context": make_context(project_id, "RUN-TEST"), "target_journal_id": None}
    create_manuscript_plan_node(state, repo)

    sections = {s.name: s.claim_ids for s in repo.get_planned_sections(project_id)}
    assert "CLAIM-RESULT" in sections["Results"]
    assert "CLAIM-FACT" in sections["Introduction"]
    assert "CLAIM-INTERP" in sections["Discussion"]
