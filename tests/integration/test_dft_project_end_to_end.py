from __future__ import annotations

"""Offline end-to-end integration test for a simulated DFT project folder.

The test intentionally uses fake LLM/provider implementations. It exercises the
real deterministic scanners, LangGraph graph construction, state contracts,
figure generation, literature grounding, drafting, review, and validation paths
without network calls or scientific-code execution.

Run from the repository root with:

    pytest -q tests/integration/test_dft_project_end_to_end.py

Ported from the repo-root test_dft_project_end_to_end.py prototype. Only the
imports changed, to match literature_figures_graph.py and
manuscript_review_graph.py moving to
src/manuscript_system/graphs/subgraphs/{literature_and_figures,drafting_and_peer_review}.py.
"""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from manuscript_system.graphs.subgraphs.literature_and_figures import (  # noqa: E402
    Claim,
    DataAsset,
    Evidence,
    FigureState,
    LiteratureRecord,
    LiteratureState,
    Repository,
    SufficiencyState,
    DFT_PROFILE,
    build_figure_graph,
    build_literature_graph,
    dft_checks,
    generate_figure,
    run_sufficiency,
)
from manuscript_system.graphs.subgraphs.drafting_and_peer_review import (  # noqa: E402
    DraftState,
    ManuscriptRepository,
    ReviewState,
    SectionSpec,
    build_drafting_graph,
    build_review_graph,
)


class OfflineLLM:
    """Schema-aware fake LLM for deterministic integration testing."""

    model = "offline-test-model"

    def call(self, system, payload, schema):
        name = schema.__name__
        if name == "GroundingAssessment":
            claim = payload["claim"]
            literature = payload["literature"]
            return schema(
                claim_id=claim["claim_id"],
                literature_id=literature["literature_id"],
                support_label="CONTEXT_ONLY",
                supporting_evidence_ids=[],
                exact_scope="The record is related to the DFT problem but does not directly prove the project result.",
                limitations=["Offline fixture provides metadata only."],
                confidence=0.72,
                requires_human_review=True,
            )
        if name == "NoveltyComparison":
            claim = payload["claim"]
            literature = payload["literature"]
            return schema(
                claim_id=claim["claim_id"],
                literature_id=literature["literature_id"],
                relationship="EXTENDS",
                similarity_type="METHOD_SIMILARITY",
                overlap_summary="Both studies evaluate a related first-principles calculation.",
                difference_summary="The fixture project evaluates a distinct convergence-controlled case.",
                evidence_ids=claim["evidence_ids"],
                confidence=0.76,
                requires_human_review=True,
            )
        if name == "DraftBlockProposal":
            spec = payload["section_spec"]
            claim_ids = [x["claim_id"] for x in payload["claims"]]
            evidence_ids = [x["evidence_id"] for x in payload["evidence"]]
            return schema(
                section_id=spec["section_id"],
                text=f"{spec['title']} reports only the approved DFT evidence within the tested scope.",
                claim_ids=claim_ids,
                evidence_ids=evidence_ids,
                literature_ids=[],
                warnings=[],
                abstain=False,
            )
        if name == "ReviewerReport":
            blocks = payload["blocks"]
            block_id = blocks[0]["block_id"] if blocks else ""
            claim_id = payload["claims"][0]["claim_id"]
            return schema(
                reviewer_id=payload["reviewer_id"],
                persona=payload["persona"],
                summary="The fixture manuscript is coherent but should state the limited convergence scope.",
                comments=[{
                    "comment_id": f"COMMENT-{payload['persona']}",
                    "reviewer_id": payload["reviewer_id"],
                    "severity": "MINOR",
                    "category": "OVERCLAIM",
                    "text": "Clarify that the result is limited to the tested convergence range.",
                    "affected_section_ids": [blocks[0]["section_id"]] if blocks else [],
                    "affected_block_ids": [block_id] if block_id else [],
                    "affected_claim_ids": [claim_id],
                    "evidence_ids": payload["claims"][0]["evidence_ids"],
                    "validity": "VALID",
                    "status": "OPEN",
                    "rationale": "The fixture claim is intentionally scoped.",
                }],
                overall_recommendation="MINOR_REVISION",
                evidence_ids=payload["claims"][0]["evidence_ids"],
                abstain=False,
            )
        if name == "RevisionProposal":
            comment = payload["comment"]
            block = payload["affected_blocks"][0]
            return schema(
                comment_id=comment["comment_id"],
                section_id=block["section_id"],
                original_block_id=block["block_id"],
                revised_text=block["text"] + " The interpretation is limited to the tested convergence range.",
                retained_claim_ids=block["claim_ids"],
                added_claim_ids=[],
                removed_claim_ids=[],
                evidence_ids=block["evidence_ids"],
                change_summary="Added the reviewer-requested scope limitation.",
                response_to_reviewer="We clarified the tested convergence scope in the revised section.",
                requires_human_review=True,
                abstain=False,
            )
        raise AssertionError(f"Unexpected schema in offline test: {name}")


class OfflineProviders:
    def search(self, request):
        return [LiteratureRecord(
            literature_id="OPENALEX:W-OFFLINE-1",
            provider="OPENALEX",
            provider_id="W-OFFLINE-1",
            title="Calculated total energy convergence within a tested cutoff range",
            authors=["Test Author"],
            year=2024,
            doi="10.0000/offline.fixture",
            journal="Fixture Journal",
            abstract="A related first-principles study reports calculated total energy convergence within a tested cutoff range.",
            source_url="https://example.org/offline-fixture",
            metadata_verified=True,
            verification_notes=["offline fixture"],
            query=request.query,
            retrieved_at="2026-08-21T00:00:00Z",
        )]


def create_dft_project(tmp_path: Path) -> tuple[Path, list[Claim], list[Evidence], list[DataAsset]]:
    project = tmp_path / "dft_project"
    project.mkdir()
    metadata = {
        "software_version": "fixture-dft-1.0",
        "functional": "PBE",
        "pseudopotential_or_basis": "fixture-PAW",
        "cutoff": 520,
        "kpoint_mesh": [4, 4, 4],
        "convergence_criteria": "1e-6 eV",
        "structure_or_molecule_id": "Si2-fixture",
        "convergence_evidence": True,
    }
    (project / "run_manifest.json").write_text(json.dumps(metadata, indent=2))
    convergence = pd.DataFrame({
        "cutoff_eV": [300, 400, 500, 520, 600],
        "total_energy_eV": [-10.10, -10.22, -10.25, -10.251, -10.2512],
    })
    convergence.to_csv(project / "convergence.csv", index=False)
    results = pd.DataFrame({
        "reference": [-10.1, -10.2, -10.3, -10.4],
        "prediction": [-10.09, -10.19, -10.31, -10.39],
    })
    results.to_csv(project / "results.csv", index=False)
    (project / "README.md").write_text("Fixture DFT project with convergence and result tables.\n")
    (project / "PSEUDOPOTENTIAL").write_text("Fixture pseudopotential; not a real scientific input.\n")

    claims = [Claim(
        claim_id="CLM-DFT-001",
        text="The calculated total energy is converged within the tested cutoff range.",
        claim_type="RESULT",
        importance="CENTRAL",
        evidence_ids=["EVD-DFT-CONV"],
        approved=True,
    )]
    evidence = [
        Evidence(
            evidence_id="EVD-DFT-CONV",
            source_artifact_id="ART-CONV",
            evidence_type="RESULT",
            location="CLM-DFT-001:convergence.csv:cutoff_eV,total_energy_eV",
            excerpt_or_value="Energy changes decrease at the high-cutoff end.",
        ),
        Evidence(
            evidence_id="EVD-DFT-META",
            source_artifact_id="ART-MANIFEST",
            evidence_type="USER_INPUT",
            location="run_manifest.json",
            excerpt_or_value=json.dumps(metadata),
        ),
    ]
    assets = [
        DataAsset(
            asset_id="ART-CONV",
            path=str(project / "convergence.csv"),
            kind="CSV",
            columns=["cutoff_eV", "total_energy_eV"],
            metadata=metadata,
        ),
        DataAsset(
            asset_id="ART-RESULTS",
            path=str(project / "results.csv"),
            kind="CSV",
            columns=["reference", "prediction"],
            metadata=metadata,
        ),
    ]
    return project, claims, evidence, assets


def test_dft_project_scanner_and_sufficiency(tmp_path):
    project, claims, evidence, assets = create_dft_project(tmp_path)
    repo = Repository()
    checks = dft_checks(DFT_PROFILE, assets, repo)
    report = run_sufficiency(DFT_PROFILE, assets, repo, "PROJECT-DFT-001")
    assert project.exists()
    assert checks
    assert report.project_id == "PROJECT-DFT-001"
    assert report.status in {"SUFFICIENT", "SUFFICIENT_WITH_WARNINGS"}
    assert report.coverage_score > 0


def test_dft_figure_generation_is_provenance_linked(tmp_path):
    project, claims, evidence, assets = create_dft_project(tmp_path)
    requirement = {
        "requirement_id": "PLOT-DFT-CONV",
        "claim_id": "CLM-DFT-001",
        "plot_type": "CONVERGENCE",
        "title": "DFT cutoff convergence fixture",
        "required_columns": ["cutoff_eV", "total_energy_eV"],
        "required_checks": ["finite_values", "units_present"],
        "importance": "CENTRAL",
    }
    from manuscript_system.graphs.subgraphs.literature_and_figures import PlotRequirement, review_figure
    figure = generate_figure(
        PlotRequirement.model_validate(requirement),
        assets[0],
        str(tmp_path / "figures"),
        "PROJECT-DFT-001",
        code_revision="test-revision",
    )
    review = review_figure(figure, PlotRequirement.model_validate(requirement))
    assert Path(figure.path).exists()
    assert figure.source_asset_ids == ["ART-CONV"]
    assert figure.claim_ids == ["CLM-DFT-001"]
    assert review.status == "PASS"


def test_literature_grounding_and_novelty_graph_offline():
    claims = [Claim(
        claim_id="CLM-DFT-001",
        text="The calculated total energy is converged within the tested cutoff range.",
        claim_type="RESULT",
        importance="CENTRAL",
        evidence_ids=["EVD-DFT-CONV"],
        approved=True,
    )]
    evidence = [Evidence(
        evidence_id="EVD-DFT-CONV",
        source_artifact_id="ART-CONV",
        evidence_type="RESULT",
        location="convergence.csv",
        excerpt_or_value="Convergence table",
    )]
    graph = build_literature_graph(OfflineProviders(), OfflineLLM())
    state: LiteratureState = {
        "project_id": "PROJECT-DFT-001",
        "claims": [x.model_dump(mode="json") for x in claims],
        "evidence": [x.model_dump(mode="json") for x in evidence],
    }
    result = graph.invoke(state)
    assert result["literature_records"]
    assert result["grounding"]
    assert result["comparisons"]
    assert result["novelty_assessments"][0]["status"] == "PARTIALLY_DISTINCT"


def test_section_drafting_graph_offline():
    llm = OfflineLLM()
    repo = ManuscriptRepository()
    graph = build_drafting_graph(llm, repo)
    claims = [{
        "claim_id": "CLM-DFT-001",
        "text": "The calculated total energy is converged within the tested cutoff range.",
        "claim_type": "RESULT",
        "importance": "CENTRAL",
        "evidence_ids": ["EVD-DFT-CONV"],
        "literature_ids": [],
        "approved": True,
    }]
    evidence = [{
        "evidence_id": "EVD-DFT-CONV",
        "source_artifact_id": "ART-CONV",
        "evidence_type": "RESULT",
        "location": "convergence.csv",
        "excerpt_or_value": "Convergence table",
    }]
    section = SectionSpec(
        section_id="SEC-RESULTS",
        title="Results",
        section_type="RESULTS",
        required_claim_ids=["CLM-DFT-001"],
        required_evidence_ids=["EVD-DFT-CONV"],
        generation_order=1,
        approved=True,
    )
    result = graph.invoke({
        "project_id": "PROJECT-DFT-001",
        "journal_profile": {"journal_id": "fixture", "version": "1", "required_sections": ["RESULTS"], "citation_style": "numeric"},
        "section_specs": [section.model_dump(mode="json")],
        "current_section_index": 0,
        "claims": claims,
        "evidence": evidence,
    })
    assert result["blocks"]
    assert result["blocks"][0]["section_id"] == "SEC-RESULTS"
    assert result["blocks"][0]["evidence_ids"] == ["EVD-DFT-CONV"]


def test_review_graph_offline_until_revision_plan():
    llm = OfflineLLM()
    repo = ManuscriptRepository()
    draft_graph = build_drafting_graph(llm, repo)
    review_graph = build_review_graph(llm, llm, repo)
    claims = [{
        "claim_id": "CLM-DFT-001",
        "text": "The calculated total energy is converged within the tested cutoff range.",
        "claim_type": "RESULT",
        "importance": "CENTRAL",
        "evidence_ids": ["EVD-DFT-CONV"],
        "literature_ids": [],
        "approved": True,
    }]
    evidence = [{
        "evidence_id": "EVD-DFT-CONV",
        "source_artifact_id": "ART-CONV",
        "evidence_type": "RESULT",
        "location": "convergence.csv",
        "excerpt_or_value": "Convergence table",
    }]
    section = SectionSpec(
        section_id="SEC-RESULTS",
        title="Results",
        section_type="RESULTS",
        required_claim_ids=["CLM-DFT-001"],
        required_evidence_ids=["EVD-DFT-CONV"],
        generation_order=1,
        approved=True,
    )
    draft_result = draft_graph.invoke({
        "project_id": "PROJECT-DFT-001",
        "journal_profile": {"journal_id": "fixture", "version": "1", "required_sections": ["RESULTS"], "citation_style": "numeric"},
        "section_specs": [section.model_dump(mode="json")],
        "current_section_index": 0,
        "claims": claims,
        "evidence": evidence,
    })
    blocks = draft_result["blocks"]
    assert blocks
    review_result = review_graph.invoke({
        "project_id": "PROJECT-DFT-001",
        "journal_profile": {"journal_id": "fixture", "version": "1", "required_sections": ["RESULTS"], "citation_style": "numeric"},
        "blocks": blocks,
        "claims": claims,
        "evidence": evidence,
        "reviewer_personas": ["SCIENTIFIC_EXPERT"],
        "reviewer_index": 0,
        "review_round": 1,
    })
    assert review_result["review_reports"]
    assert review_result["revision_plan"]["comment_ids"]


def test_end_to_end_dft_walkthrough(tmp_path):
    """High-level acceptance test for the complete offline workflow."""
    project, claims, evidence, assets = create_dft_project(tmp_path)
    repo = Repository()
    report = run_sufficiency(DFT_PROFILE, assets, repo, "PROJECT-DFT-001")
    assert report.status != "INSUFFICIENT"

    figure_requirement = {
        "requirement_id": "PLOT-DFT-CONV",
        "claim_id": "CLM-DFT-001",
        "plot_type": "CONVERGENCE",
        "title": "DFT cutoff convergence fixture",
        "required_columns": ["cutoff_eV", "total_energy_eV"],
        "required_checks": ["finite_values", "units_present"],
        "importance": "CENTRAL",
    }
    from manuscript_system.graphs.subgraphs.literature_and_figures import PlotRequirement
    figure = generate_figure(
        PlotRequirement.model_validate(figure_requirement),
        assets[0],
        str(tmp_path / "figures"),
        "PROJECT-DFT-001",
        code_revision="test-revision",
    )
    assert figure.status == "PROPOSED"
    assert Path(figure.path).exists()

    literature_graph = build_literature_graph(OfflineProviders(), OfflineLLM())
    lit_result = literature_graph.invoke({
        "project_id": "PROJECT-DFT-001",
        "claims": [x.model_dump(mode="json") for x in claims],
        "evidence": [x.model_dump(mode="json") for x in evidence],
    })
    assert lit_result["novelty_assessments"]

    draft_graph = build_drafting_graph(OfflineLLM(), ManuscriptRepository())
    section = SectionSpec(
        section_id="SEC-RESULTS",
        title="Results",
        section_type="RESULTS",
        required_claim_ids=["CLM-DFT-001"],
        required_evidence_ids=["EVD-DFT-CONV"],
        generation_order=1,
        approved=True,
    )
    draft = draft_graph.invoke({
        "project_id": "PROJECT-DFT-001",
        "journal_profile": {"journal_id": "fixture", "version": "1", "required_sections": ["RESULTS"], "citation_style": "numeric"},
        "section_specs": [section.model_dump(mode="json")],
        "current_section_index": 0,
        "claims": [x.model_dump(mode="json") for x in claims],
        "evidence": [x.model_dump(mode="json") for x in evidence],
        "literature": lit_result.get("literature_records", []),
    })
    assert draft["blocks"]
    assert draft["blocks"][0]["claim_ids"] == ["CLM-DFT-001"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
