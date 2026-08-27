from __future__ import annotations

import os
from typing import Any

from ..domain.models import Claim, Evidence, ManuscriptBlock, PlannedSection
from ..persistence.repositories import Repository
from .evidence_extraction import _CLAIM_TYPE_MAP
from .subgraphs.drafting_and_peer_review import (
    Claim as PortedClaim,
    DraftState,
    Evidence as PortedEvidence,
    ManuscriptBlock as PortedManuscriptBlock,
    SectionSpec,
    StructuredLLM,
    build_drafting_graph,
)

"""Wires drafting_and_peer_review.py's write_next_section_node (real LLM
calls, one per section) into the real system - assessment.py's
create_manuscript_plan_node already produces an approved ManuscriptPlan with
PlannedSections and claim allocations; this turns that into a drafted
ManuscriptBlock per section.

Only drafting is wired here, not the 5-persona peer-review/revision cycle in
the same ported module - that's 15-20+ LLM calls per cycle through a much
bigger adapter surface, deferred to a session with a fresh LLM quota.

Real constraint (checked against the actual AQT ManuscriptPlan): only
sections with at least one allocated claim get drafted. evidence_extraction.py
currently only ever emits claim_type="RESULT" claims, so most IMRaD sections
(Introduction, Methods, Conclusion) have zero claims - WRITER_SYSTEM
instructs the model to abstain rather than invent content, and an abstained
section sends the whole graph straight to BLOCKED with no continuing to
later sections. Skipping empty sections up front avoids drafting hitting an
empty Introduction first and blocking before ever reaching Results.
"""

DRAFT_CHOICES = (
    "APPROVE_DRAFT_FOR_PEER_REVIEW",
    "REQUEST_DRAFT_REVISION",
    "BLOCK_DRAFT",
)

_SECTION_TYPE_KEYWORDS = [
    ("introduction", "INTRODUCTION"),
    ("method", "METHODS"),
    ("result", "RESULTS"),
    ("discussion", "DISCUSSION"),
    ("conclusion", "CONCLUSION"),
    ("abstract", "ABSTRACT"),
    ("title", "TITLE"),
    ("data availability", "DATA_AVAILABILITY"),
    ("code availability", "CODE_AVAILABILITY"),
    ("acknowledgement", "ACKNOWLEDGEMENTS"),
    ("conflict", "CONFLICTS"),
    ("reference", "REFERENCES"),
]

_EVIDENCE_TYPE_MAP = {
    "USER_INPUT": "USER_INPUT", "EXPERIMENTAL_RESULT": "RESULT", "COMPUTATIONAL_RESULT": "RESULT",
    "LITERATURE": "LITERATURE", "FIGURE": "FIGURE", "TABLE": "TABLE", "CODE": "CODE", "INFERENCE": "RESULT",
}


def _map_section_type(name: str) -> str:
    lower = name.lower()
    for keyword, section_type in _SECTION_TYPE_KEYWORDS:
        if keyword in lower:
            return section_type
    return "DISCUSSION"


def _to_ported_claim(claim: Claim) -> PortedClaim:
    return PortedClaim(
        claim_id=claim.claim_id, text=claim.text,
        claim_type=_CLAIM_TYPE_MAP.get(claim.claim_type, "RESULT"),
        importance=claim.importance, evidence_ids=claim.evidence_ids, literature_ids=[], approved=True,
    )


def _to_ported_evidence(item: Evidence) -> PortedEvidence:
    return PortedEvidence(
        evidence_id=item.evidence_id, source_artifact_id=item.source_artifact_id,
        evidence_type=_EVIDENCE_TYPE_MAP.get(item.evidence_type, "RESULT"),
        location=item.location, excerpt_or_value=item.excerpt_or_value, units=None,
    )


def build_section_specs(
    sections: list[PlannedSection], claims: list[Claim],
) -> list[SectionSpec]:
    """Only sections with at least one allocated claim - see module docstring."""
    claims_by_id = {c.claim_id: c for c in claims}
    specs: list[SectionSpec] = []
    for section in sorted(sections, key=lambda s: s.order):
        if not section.claim_ids:
            continue
        evidence_ids: list[str] = []
        for claim_id in section.claim_ids:
            claim = claims_by_id.get(claim_id)
            if claim:
                evidence_ids.extend(claim.evidence_ids)
        specs.append(SectionSpec(
            section_id=section.section_id, title=section.name, section_type=_map_section_type(section.name),
            required_claim_ids=section.claim_ids, required_evidence_ids=evidence_ids,
            generation_order=section.order, approved=True,
        ))
    return specs


class _RepoAdapter:
    """Satisfies write_next_section_node's repo.put_block(block) call (that
    function is reused unmodified from drafting_and_peer_review.py) by
    converting the ported ManuscriptBlock it constructs into the real,
    persisted one."""

    def __init__(self, repo: Repository, project_id: str | None, plan_id: str | None) -> None:
        self._repo = repo
        self._project_id = project_id
        self._plan_id = plan_id

    def put_block(self, block: PortedManuscriptBlock) -> str:
        self._repo.add_manuscript_block(ManuscriptBlock(
            block_id=block.block_id, project_id=self._project_id, plan_id=self._plan_id,
            section_id=block.section_id, text=block.text, claim_ids=block.claim_ids,
            evidence_ids=block.evidence_ids, literature_ids=block.literature_ids,
            authoring_agent=block.authoring_agent, prompt_hash=block.prompt_hash, model_id=block.model_id,
            code_revision=block.code_revision, status=block.status, semantic_warnings=block.semantic_warnings,
        ))
        return block.block_id


def initial_state(
    *, project_id: str, section_specs: list[SectionSpec], claims: list[Claim], evidence: list[Evidence],
) -> DraftState:
    return {
        "project_id": project_id,
        "journal_profile": {},
        "section_specs": [s.model_dump(mode="json") for s in section_specs],
        "current_section_index": 0,
        "claims": [_to_ported_claim(c).model_dump(mode="json") for c in claims],
        "evidence": [_to_ported_evidence(e).model_dump(mode="json") for e in evidence],
        "literature": [],
        "blocks": [],
        "block_ids": [],
        "draft_status": "RUNNING",
    }


def build_manuscript_drafting_graph(
    repo: Repository, project_id: str | None = None, plan_id: str | None = None,
    checkpoint_db_path: str | None = None,
):
    """project_id/plan_id are only used if write_next_section actually runs
    (drafting a new section) - not needed to resume an existing thread past
    human_draft_review_node's interrupt, since that never calls put_block again."""
    llm = StructuredLLM(os.getenv("DRAFTING_MODEL", "gpt-5-mini"))
    repo_adapter = _RepoAdapter(repo, project_id, plan_id)
    return build_drafting_graph(llm, repo_adapter, checkpoint_db_path)
