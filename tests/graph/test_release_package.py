from __future__ import annotations

"""Tests for tools/release_package.py's assemble_release_package -
deterministic, no LLM/network. Covers a project with a full set of real
records present, and a bare project with almost nothing recorded (must
render honestly rather than erroring), plus a package_hash sanity check.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from manuscript_system.domain.models import (  # noqa: E402
    Claim,
    CompletionPlan,
    CompletionTask,
    Evidence,
    Finding,
    HumanDecision,
    ManuscriptBlock,
    ManuscriptPlan,
    PlannedSection,
    ReadinessReport,
)
from manuscript_system.persistence.database import connect  # noqa: E402
from manuscript_system.persistence.repositories import Repository, new_id  # noqa: E402
from manuscript_system.tools.release_package import assemble_release_package  # noqa: E402


def make_repo(tmp_path: Path) -> Repository:
    return Repository(connect(str(tmp_path / "test.sqlite")))


def test_assemble_release_package_with_full_data(tmp_path):
    repo = make_repo(tmp_path)
    project = repo.create_project(source_path="/fake/source", domain_profile_id="quantum_chemistry")
    project_id = project.project_id

    evidence = Evidence(
        evidence_id="EVIDENCE-1", project_id=project_id, source_artifact_id="ARTIFACT-1",
        evidence_type="TABLE", location="CLAIM-1:x.csv", excerpt_or_value="raw data excerpt",
    )
    repo.add_evidence_items([evidence])
    claim = Claim(
        claim_id="CLAIM-1", project_id=project_id, text="A central result", claim_type="RESULT",
        importance="CENTRAL", evidence_ids=[evidence.evidence_id],
    )
    repo.add_claims([claim])

    plan_id = "PLAN-1"
    section_drafted = PlannedSection(
        section_id="SECTION-1", project_id=project_id, plan_id=plan_id, name="Results",
        order=0, claim_ids=[claim.claim_id],
    )
    section_empty = PlannedSection(
        section_id="SECTION-2", project_id=project_id, plan_id=plan_id, name="Introduction",
        order=1, claim_ids=[],
    )
    repo.add_planned_sections([section_drafted, section_empty])
    repo.add_manuscript_plan(ManuscriptPlan(
        plan_id=plan_id, project_id=project_id, run_id="RUN-1",
        section_ids=[section_drafted.section_id, section_empty.section_id],
    ))

    repo.add_manuscript_block(ManuscriptBlock(
        block_id="BLOCK-1", project_id=project_id, plan_id=plan_id, section_id=section_drafted.section_id,
        text="The result was demonstrated.", claim_ids=[claim.claim_id], evidence_ids=[evidence.evidence_id],
    ))

    repo.add_readiness_report(ReadinessReport(
        report_id="REPORT-1", project_id=project_id, run_id="RUN-1",
        readiness_status="DRAFTABLE_WITH_WARNINGS", explanation="Some evidence gaps remain.",
    ))

    task = CompletionTask(
        task_id="TASK-1", project_id=project_id, plan_id="COMPPLAN-1", title="Add benchmark",
        reason="No benchmark provided.", category="VALIDATION", priority="REQUIRED",
    )
    repo.add_completion_tasks([task])
    repo.add_completion_plan(CompletionPlan(plan_id="COMPPLAN-1", project_id=project_id, run_id="RUN-1", task_ids=[task.task_id]))

    repo.add_finding(Finding(
        finding_id="FINDING-1", project_id=project_id, severity="MEDIUM",
        rule_id="missing_readme", message="No README file found.",
    ))
    repo.add_human_decision(HumanDecision(
        decision_id="DECISION-1", project_id=project_id, run_id="RUN-1",
        kind="ASSESSMENT_REVIEW", decision="APPROVE_MANUSCRIPT_PLANNING",
    ))

    candidate, markdown = assemble_release_package(repo, project_id, str(tmp_path / "checkpoints.sqlite"))

    assert candidate.project_id == project_id
    assert candidate.manuscript_plan_id == plan_id
    assert candidate.section_count == 2
    assert candidate.drafted_section_count == 1
    assert candidate.human_decision_count == 1
    assert candidate.package_hash

    assert "/fake/source" in markdown
    assert "DRAFTABLE_WITH_WARNINGS" in markdown
    assert "The result was demonstrated." in markdown
    assert "not drafted" in markdown  # Introduction section
    assert "Add benchmark" in markdown
    assert "missing_readme" in markdown
    assert "APPROVE_MANUSCRIPT_PLANNING" in markdown
    assert "no citations recorded" in markdown  # honest, not faked


def test_assemble_release_package_with_almost_nothing(tmp_path):
    """A bare project (no plan, no readiness, no completion plan, no peer
    review, no citations, no findings, no decisions) must render honestly
    rather than raising."""
    repo = make_repo(tmp_path)
    project_id = "PROJECT-BARE"

    candidate, markdown = assemble_release_package(repo, project_id, str(tmp_path / "checkpoints.sqlite"))

    assert candidate.project_id == project_id
    assert candidate.manuscript_plan_id is None
    assert candidate.section_count == 0
    assert candidate.drafted_section_count == 0
    assert "no approved manuscript plan" in markdown
    assert "no readiness report" in markdown
    assert "no completion plan" in markdown
    assert "peer review has not been run" in markdown
    assert "no citations recorded" in markdown
    assert "no findings recorded" in markdown
    assert "no human decisions recorded" in markdown


def test_package_hash_changes_when_content_changes(tmp_path):
    repo = make_repo(tmp_path)
    candidate1, _ = assemble_release_package(repo, "PROJECT-A", str(tmp_path / "checkpoints.sqlite"))

    repo.add_finding(Finding(
        finding_id="FINDING-1", project_id="PROJECT-A", severity="LOW",
        rule_id="missing_environment_file", message="No environment file found.",
    ))
    candidate2, _ = assemble_release_package(repo, "PROJECT-A", str(tmp_path / "checkpoints.sqlite"))

    assert candidate1.package_hash != candidate2.package_hash
