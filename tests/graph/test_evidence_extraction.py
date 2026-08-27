from __future__ import annotations

"""Tests for evidence_extraction.py and its wiring into assessment.py's
run_audits. The zero-claims path must stay LLM-free (same discipline as
Phase 0); the real-evidence path is skipped unless OPENAI_API_KEY is set,
since it makes a genuine call through model_gateway.
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import pytest  # noqa: E402

from manuscript_system.domain.models import SourceAsset  # noqa: E402
from manuscript_system.graphs.assessment import build_assessment_graph, initial_state  # noqa: E402
from manuscript_system.graphs.evidence_extraction import (  # noqa: E402
    classify_domain_from_assets,
    extract_evidence_and_claims,
    load_domain_profile,
)
from manuscript_system.persistence.database import connect  # noqa: E402
from manuscript_system.persistence.repositories import Repository  # noqa: E402


def make_repo(tmp_path: Path) -> Repository:
    return Repository(connect(str(tmp_path / "test.sqlite")))


def make_asset(relative_path: str, project_id: str = "P") -> SourceAsset:
    return SourceAsset(
        artifact_id=f"ASSET-{relative_path}", project_id=project_id, relative_path=relative_path,
        checksum_sha256="x" * 8, size_bytes=1, media_type="application/octet-stream",
    )


def test_classify_domain_from_assets_no_match_falls_back_to_generic():
    assets = [make_asset("notes.md"), make_asset("results.csv")]
    domain_id, confidence = classify_domain_from_assets(assets)
    assert domain_id == "generic"
    assert confidence == 0.0


def test_classify_domain_from_assets_matches_quantum_chemistry():
    assets = [
        make_asset("data/raw/dissociation_curves/lipf6_ccpvtz_dissociation.dat"),
        make_asset("scripts/casci_benchmark.py"),
    ]
    domain_id, confidence = classify_domain_from_assets(assets)
    assert domain_id == "quantum_chemistry"
    assert confidence > 0


def test_extract_evidence_and_claims_no_domain_profile_produces_nothing():
    claims, evidence, findings = extract_evidence_and_claims([], None, "P")
    assert claims == evidence == findings == []


def test_extract_evidence_and_claims_missing_checklist_item_produces_finding():
    domain_profile = load_domain_profile("quantum_chemistry")
    claims, evidence, findings = extract_evidence_and_claims([], domain_profile, "PROJECT-X")
    assert claims == []
    assert evidence == []
    assert len(findings) == len(domain_profile["completeness_checklist"])
    assert all(not f.blocking for f in findings)


def test_extract_evidence_and_claims_matches_real_checklist_items():
    domain_profile = load_domain_profile("quantum_chemistry")
    assets = [
        make_asset("data/raw/dissociation_curves/lipf6_ccpvtz_dissociation.dat"),
        make_asset("data/raw/basis_convergence/NaPF6_dissociation_absolute.csv"),
    ]
    claims, evidence, findings = extract_evidence_and_claims(assets, domain_profile, "PROJECT-X")
    assert len(claims) >= 1
    assert all(claim.evidence_ids for claim in claims)
    for claim in claims:
        for evidence_id in claim.evidence_ids:
            item = next(e for e in evidence if e.evidence_id == evidence_id)
            # The claim_id-prefixed location convention evaluate_blockers/
            # compute_empirical_features rely on (novelty_and_publishability.py).
            assert item.location.startswith(claim.claim_id + ":")


def test_run_audits_with_no_claims_makes_zero_llm_calls(tmp_path):
    (tmp_path / "readme.txt").write_text("nothing domain-specific here")
    repo = make_repo(tmp_path)
    graph = build_assessment_graph(repo, str(tmp_path / "checkpoints.sqlite"))
    thread_id = "thread-extraction-empty"
    config = {"configurable": {"thread_id": thread_id}}
    state = initial_state(
        source_path=str(tmp_path), project_id="PROJECT-EMPTY", run_id="RUN-EMPTY", thread_id=thread_id,
    )
    result = graph.invoke(state, config=config)
    assert result["__interrupt__"]
    assert result["candidate_claim_ids"] == []
    assert result["readiness_status"] in {"DRAFTABLE_WITH_WARNINGS", "NEEDS_ADDITIONAL_ANALYSIS"}


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="requires a configured LLM provider")
def test_run_audits_with_real_evidence_produces_llm_readiness(tmp_path):
    (tmp_path / "dissociation_curve.dat").write_text("cutoff,energy\n1,2\n")
    (tmp_path / "basis_convergence.csv").write_text("basis,energy\ncc-pVDZ,1\ncc-pVTZ,2\n")
    repo = make_repo(tmp_path)
    graph = build_assessment_graph(repo, str(tmp_path / "checkpoints.sqlite"))
    thread_id = "thread-extraction-real"
    config = {"configurable": {"thread_id": thread_id}}
    state = initial_state(
        source_path=str(tmp_path), project_id="PROJECT-REAL", run_id="RUN-REAL", thread_id=thread_id,
    )
    result = graph.invoke(state, config=config)
    assert result["__interrupt__"]
    assert result["candidate_claim_ids"]
    assert result["readiness_status"] in {
        "READY_FOR_MANUSCRIPT", "DRAFTABLE_WITH_WARNINGS", "NEEDS_ADDITIONAL_ANALYSIS",
        "INSUFFICIENT_EVIDENCE", "BLOCKED",
    }
    rows = repo.conn.execute(
        "SELECT claim_id FROM claims WHERE project_id = ?", ("PROJECT-REAL",)
    ).fetchall()
    assert rows
