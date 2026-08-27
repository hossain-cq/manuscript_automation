from __future__ import annotations

"""Tests for graphs/manuscript_drafting.py.

Adapter/mapping tests are fully synthetic (no network). The one real test
makes a real LLM call and is skipped without a configured provider - same
convention as test_evidence_extraction.py's
test_run_audits_with_real_evidence_produces_llm_readiness. It may still fail
if the provider's quota is exhausted; that's an environmental condition, not
a code bug, and is checked separately before relying on this test.
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import pytest  # noqa: E402

from manuscript_system.domain.models import Claim, Evidence, PlannedSection  # noqa: E402
from manuscript_system.graphs.manuscript_drafting import (  # noqa: E402
    _map_section_type,
    _to_ported_claim,
    _to_ported_evidence,
    build_manuscript_drafting_graph,
    build_section_specs,
    initial_state,
)
from manuscript_system.persistence.database import connect  # noqa: E402
from manuscript_system.persistence.repositories import Repository, new_id  # noqa: E402


def make_repo(tmp_path: Path) -> Repository:
    return Repository(connect(str(tmp_path / "test.sqlite")))


# --- section type mapping -----------------------------------------------------


def test_map_section_type_combined_results_and_discussion_prefers_results():
    """Real AQT journal profile combines these into one section name."""
    assert _map_section_type("Results And Discussion") == "RESULTS"


def test_map_section_type_basic_cases():
    assert _map_section_type("Introduction") == "INTRODUCTION"
    assert _map_section_type("Methodology") == "METHODS"
    assert _map_section_type("Conclusion") == "CONCLUSION"
    assert _map_section_type("Abstract") == "ABSTRACT"


def test_map_section_type_unknown_name_falls_back_to_discussion():
    assert _map_section_type("Something Unusual") == "DISCUSSION"


# --- claim/evidence adapters ---------------------------------------------------


def test_to_ported_claim_maps_fact_and_suggestion():
    fact = Claim(claim_id="C1", project_id="P1", text="x", claim_type="FACT", importance="CENTRAL")
    suggestion = Claim(claim_id="C2", project_id="P1", text="x", claim_type="SUGGESTION", importance="SUPPORTING")
    assert _to_ported_claim(fact).claim_type == "RESULT"
    assert _to_ported_claim(suggestion).claim_type == "INTERPRETATION"
    assert _to_ported_claim(fact).approved is True


def test_to_ported_evidence_maps_computational_result():
    """evidence_extraction.py's own fallback for unmatched extensions -
    confirmed this is a real value the adapter must handle, not just
    defensive code."""
    item = Evidence(
        evidence_id="E1", project_id="P1", source_artifact_id="A1",
        evidence_type="COMPUTATIONAL_RESULT", location="C1:x.txt", excerpt_or_value="...",
    )
    assert _to_ported_evidence(item).evidence_type == "RESULT"


def test_to_ported_evidence_passes_through_known_types():
    for evidence_type in ("FIGURE", "TABLE", "CODE"):
        item = Evidence(
            evidence_id="E1", project_id="P1", source_artifact_id="A1",
            evidence_type=evidence_type, location="C1:x", excerpt_or_value="...",
        )
        assert _to_ported_evidence(item).evidence_type == evidence_type


# --- section spec construction -------------------------------------------------


def test_build_section_specs_skips_sections_with_no_claims():
    """Real constraint: an empty section would abstain and block the whole
    graph, so it must never be handed to the drafting graph at all."""
    claim = Claim(claim_id="CLAIM-1", project_id="P1", text="x", claim_type="RESULT", importance="CENTRAL",
                  evidence_ids=["EVIDENCE-1"])
    sections = [
        PlannedSection(section_id="S1", project_id="P1", plan_id="PLAN1", name="Introduction", order=0, claim_ids=[]),
        PlannedSection(section_id="S2", project_id="P1", plan_id="PLAN1", name="Methodology", order=1, claim_ids=[]),
        PlannedSection(
            section_id="S3", project_id="P1", plan_id="PLAN1", name="Results And Discussion", order=2,
            claim_ids=["CLAIM-1"],
        ),
        PlannedSection(section_id="S4", project_id="P1", plan_id="PLAN1", name="Conclusion", order=3, claim_ids=[]),
    ]
    specs = build_section_specs(sections, [claim])
    assert len(specs) == 1
    assert specs[0].section_id == "S3"
    assert specs[0].section_type == "RESULTS"
    assert specs[0].required_claim_ids == ["CLAIM-1"]
    assert specs[0].required_evidence_ids == ["EVIDENCE-1"]


def test_build_section_specs_returns_empty_when_no_section_has_claims():
    sections = [
        PlannedSection(section_id="S1", project_id="P1", plan_id="PLAN1", name="Introduction", order=0, claim_ids=[]),
    ]
    assert build_section_specs(sections, []) == []


# --- initial_state / graph construction (structural only, no LLM call) --------


def test_initial_state_shape():
    claim = Claim(claim_id="CLAIM-1", project_id="P1", text="x", claim_type="RESULT", importance="CENTRAL",
                  evidence_ids=["EVIDENCE-1"])
    evidence = Evidence(evidence_id="EVIDENCE-1", project_id="P1", source_artifact_id="A1",
                         evidence_type="TABLE", location="CLAIM-1:x.csv", excerpt_or_value="...")
    sections = [PlannedSection(section_id="S1", project_id="P1", plan_id="PLAN1", name="Results", order=0,
                                claim_ids=["CLAIM-1"])]
    specs = build_section_specs(sections, [claim])
    state = initial_state(project_id="P1", section_specs=specs, claims=[claim], evidence=[evidence])
    assert state["project_id"] == "P1"
    assert state["draft_status"] == "RUNNING"
    assert state["current_section_index"] == 0
    assert len(state["section_specs"]) == 1
    assert state["claims"][0]["claim_id"] == "CLAIM-1"
    assert state["evidence"][0]["evidence_id"] == "EVIDENCE-1"


def test_build_manuscript_drafting_graph_compiles_without_project_id(tmp_path):
    """approve-draft resumes a thread without needing project_id/plan_id -
    put_block is never called again past the human-review interrupt."""
    repo = make_repo(tmp_path)
    graph = build_manuscript_drafting_graph(repo, checkpoint_db_path=str(tmp_path / "checkpoints.sqlite"))
    assert graph is not None


# --- real LLM call (skipped without a configured provider) --------------------


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="requires a configured LLM provider")
def test_draft_real_section_only_references_supplied_claims_and_evidence(tmp_path):
    repo = make_repo(tmp_path)
    project_id = "PROJECT-DRAFT-REAL"
    claim = Claim(
        claim_id=new_id("CLAIM"), project_id=project_id, text="The dissociation energy of LiPF6 was computed.",
        claim_type="RESULT", importance="CENTRAL", evidence_ids=[],
    )
    evidence_item = Evidence(
        evidence_id=new_id("EVIDENCE"), project_id=project_id, source_artifact_id=new_id("ARTIFACT"),
        evidence_type="TABLE", location=f"{claim.claim_id}:dissociation_curves/lipf6.dat",
        excerpt_or_value="Distance, energy pairs for the LiPF6 dissociation curve.",
    )
    claim = claim.model_copy(update={"evidence_ids": [evidence_item.evidence_id]})
    repo.add_claims([claim])
    repo.add_evidence_items([evidence_item])

    section = PlannedSection(
        section_id=new_id("SECTION"), project_id=project_id, plan_id="PLAN-TEST", name="Results And Discussion",
        order=0, claim_ids=[claim.claim_id],
    )
    specs = build_section_specs([section], [claim])
    assert specs

    graph = build_manuscript_drafting_graph(
        repo, project_id, "PLAN-TEST", str(tmp_path / "checkpoints.sqlite"),
    )
    state = initial_state(project_id=project_id, section_specs=specs, claims=[claim], evidence=[evidence_item])
    config = {"configurable": {"thread_id": "thread-draft-real"}}
    result = graph.invoke(state, config=config)

    assert result.get("draft_status") != "BLOCKED", result.get("error")
    blocks = repo.get_manuscript_blocks(project_id)
    assert len(blocks) == 1
    block = blocks[0]
    assert set(block.claim_ids).issubset({claim.claim_id})
    assert set(block.evidence_ids).issubset({evidence_item.evidence_id})
