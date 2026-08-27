from __future__ import annotations

"""Tests for graphs/manuscript_review.py.

Adapter/mapping tests are fully synthetic (no network). The one real test
makes real LLM calls (5 reviewer personas) and is skipped without a
configured provider - same convention as test_manuscript_drafting.py's real
test. It may still fail if the provider's quota is exhausted; that's an
environmental condition, not a code bug.
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import pytest  # noqa: E402

from manuscript_system.domain.models import Claim, Evidence, ManuscriptBlock  # noqa: E402
from manuscript_system.graphs.manuscript_review import (  # noqa: E402
    _to_ported_block,
    _to_ported_journal_profile,
    build_manuscript_review_graph,
    initial_state,
)
from manuscript_system.graphs.subgraphs.drafting_and_peer_review import (  # noqa: E402
    ManuscriptRepository,
    ReviewComment,
    ReviewerReport,
    reviewer_node,
)
from manuscript_system.persistence.repositories import new_id  # noqa: E402


class _FakeLLM:
    """Stands in for StructuredLLM - returns pre-built reports in call order,
    without touching the network."""

    def __init__(self, reports: list[ReviewerReport]) -> None:
        self._reports = reports
        self._calls = 0

    def call(self, system: str, payload: dict, schema: type) -> ReviewerReport:
        report = self._reports[self._calls]
        self._calls += 1
        return report


def _make_report(persona: str, comment_id: str) -> ReviewerReport:
    return ReviewerReport(
        reviewer_id=f"reviewer-{persona.lower()}", persona=persona, summary="x",
        comments=[ReviewComment(
            comment_id=comment_id, reviewer_id=f"reviewer-{persona.lower()}", severity="MAJOR",
            category="EVIDENCE", text="issue", affected_section_ids=[], affected_block_ids=["BLOCK-1"],
            affected_claim_ids=[], evidence_ids=[], validity="VALID", status="OPEN",
        )],
        overall_recommendation="MINOR_REVISION", evidence_ids=[], abstain=False,
    )


def test_reviewer_node_namespaces_comment_ids_to_avoid_collision():
    """Regression test for a real bug found in the first live 5-persona run:
    different reviewer personas independently default to the same short
    comment_id scheme (COMMENT-1, COMMENT-2, ...), and
    ManuscriptRepository.comments is a flat dict keyed by comment_id - without
    namespacing, a later persona's COMMENT-1 silently overwrote an earlier
    persona's COMMENT-1, losing that reviewer's comment entirely."""
    llm = _FakeLLM([
        _make_report("SCIENTIFIC_EXPERT", "COMMENT-1"),
        _make_report("CRITICAL_REVIEWER", "COMMENT-1"),
    ])
    repo = ManuscriptRepository()
    base_state = {
        "reviewer_personas": ["SCIENTIFIC_EXPERT", "CRITICAL_REVIEWER"],
        "journal_profile": {}, "blocks": [], "claims": [], "evidence": [],
    }

    result1 = reviewer_node({**base_state, "reviewer_index": 0}, llm, repo)
    result2 = reviewer_node({**base_state, "reviewer_index": 1}, llm, repo)

    assert len(repo.comments) == 2  # both survive - neither overwrote the other
    all_ids = result1["comment_ids"] + result2["comment_ids"]
    assert len(set(all_ids)) == 2  # globally unique despite identical raw ids


def test_to_ported_block_maps_fields():
    block = ManuscriptBlock(
        block_id="BLOCK-1", project_id="P1", plan_id="PLAN-1", section_id="SECTION-1",
        text="Some drafted text.", claim_ids=["CLAIM-1"], evidence_ids=["EVIDENCE-1"],
        authoring_agent="writer.results", prompt_hash="abc", model_id="openai/gpt-oss-120b",
        code_revision="unknown", status="PROPOSED",
    )
    ported = _to_ported_block(block)
    assert ported.block_id == "BLOCK-1"
    assert ported.section_id == "SECTION-1"
    assert ported.text == "Some drafted text."
    assert ported.claim_ids == ["CLAIM-1"]
    assert ported.evidence_ids == ["EVIDENCE-1"]
    assert ported.status == "PROPOSED"


def test_to_ported_journal_profile_none_without_journal_id():
    assert _to_ported_journal_profile(None, {"manuscript_structure": {}}) is None


def test_to_ported_journal_profile_none_without_profile():
    assert _to_ported_journal_profile("nature", None) is None


def test_to_ported_journal_profile_maps_real_yaml_shape():
    """Real configs/journals.yaml profiles have no abstract_word_limit /
    manuscript_word_limit / required_statements fields at all - confirming
    those stay at their defaults rather than raising."""
    profile = {
        "display_name": "Advanced Quantum Technologies", "profile_version": "2026-08",
        "manuscript_structure": {"required_sections": ["title", "abstract", "results_and_discussion"]},
        "citation": {"style_id": "aps-prl"},
    }
    ported = _to_ported_journal_profile("advanced_quantum_technologies", profile)
    assert ported is not None
    assert ported.journal_id == "advanced_quantum_technologies"
    assert ported.version == "2026-08"
    assert ported.required_sections == ["title", "abstract", "results_and_discussion"]
    assert ported.citation_style == "aps-prl"
    assert ported.abstract_word_limit is None
    assert ported.required_statements == []


def test_initial_state_shape():
    block = ManuscriptBlock(
        block_id="BLOCK-1", project_id="P1", plan_id="PLAN-1", section_id="SECTION-1",
        text="Draft text.", claim_ids=["CLAIM-1"], evidence_ids=["EVIDENCE-1"],
    )
    claim = Claim(claim_id="CLAIM-1", project_id="P1", text="x", claim_type="RESULT", importance="CENTRAL",
                  evidence_ids=["EVIDENCE-1"])
    evidence = Evidence(evidence_id="EVIDENCE-1", project_id="P1", source_artifact_id="A1",
                         evidence_type="TABLE", location="CLAIM-1:x.csv", excerpt_or_value="...")
    state = initial_state(project_id="P1", blocks=[block], claims=[claim], evidence=[evidence], journal_profile=None)
    assert state["project_id"] == "P1"
    assert state["journal_profile"] == {}
    assert len(state["blocks"]) == 1
    assert state["blocks"][0]["block_id"] == "BLOCK-1"
    assert state["reviewer_index"] == 0
    assert state["terminal_status"] == "RUNNING"
    assert len(state["reviewer_personas"]) == 5


def test_build_manuscript_review_graph_compiles(tmp_path):
    graph = build_manuscript_review_graph(str(tmp_path / "checkpoints.sqlite"))
    assert graph is not None


# --- real LLM call (skipped without a configured provider) --------------------


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="requires a configured LLM provider")
def test_review_real_block_produces_five_persona_reports(tmp_path):
    project_id = "PROJECT-REVIEW-REAL"
    claim = Claim(
        claim_id=new_id("CLAIM"), project_id=project_id, text="The dissociation energy of LiPF6 was computed.",
        claim_type="RESULT", importance="CENTRAL", evidence_ids=[new_id("EVIDENCE")],
    )
    evidence_item = Evidence(
        evidence_id=claim.evidence_ids[0], project_id=project_id, source_artifact_id=new_id("ARTIFACT"),
        evidence_type="TABLE", location=f"{claim.claim_id}:dissociation_curves/lipf6.dat",
        excerpt_or_value="Distance, energy pairs for the LiPF6 dissociation curve.",
    )
    block = ManuscriptBlock(
        block_id=new_id("BLOCK"), project_id=project_id, plan_id="PLAN-TEST", section_id="SECTION-TEST",
        text=(
            "The ground-state dissociation curve for LiPF6 was computed and benchmarked against classical "
            f"CASCI calculations, with the raw data recorded in evidence {evidence_item.evidence_id} "
            f"(supporting claim {claim.claim_id})."
        ),
        claim_ids=[claim.claim_id], evidence_ids=[evidence_item.evidence_id],
    )

    graph = build_manuscript_review_graph(str(tmp_path / "checkpoints.sqlite"))
    state = initial_state(
        project_id=project_id, blocks=[block], claims=[claim], evidence=[evidence_item], journal_profile=None,
    )
    config = {"configurable": {"thread_id": "thread-review-real"}}
    result = graph.invoke(state, config=config)

    reports = result.get("review_reports", [])
    assert len(reports) == 5
    personas = {r["persona"] for r in reports}
    assert personas == {
        "SCIENTIFIC_EXPERT", "CRITICAL_REVIEWER", "STATISTICAL_COMPUTATIONAL", "JOURNAL_REVIEWER", "HOSTILE_REVIEWER",
    }
    for report in reports:
        for comment in report["comments"]:
            assert set(comment["affected_block_ids"]).issubset({block.block_id})
    assert result.get("terminal_status") == "WAITING_FOR_HUMAN"
    assert result.get("revision_plan")
