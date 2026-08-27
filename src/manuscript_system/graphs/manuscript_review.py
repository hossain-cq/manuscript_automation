from __future__ import annotations

import os
from typing import Any

from ..domain.models import Claim, Evidence, ManuscriptBlock
from .manuscript_drafting import _to_ported_claim, _to_ported_evidence
from .subgraphs.drafting_and_peer_review import (
    PERSONAS,
    JournalProfile,
    ManuscriptBlock as PortedManuscriptBlock,
    ManuscriptRepository,
    ReviewState,
    StructuredLLM,
    build_review_graph,
)

"""Wires drafting_and_peer_review.py's ReviewState graph (5 reviewer
personas -> triage -> revise -> verify -> response-to-reviewers, 3 human
gates) into the real system, against the ManuscriptBlocks manuscript_drafting.py
already produced.

Deliberately runs against the ported module's own in-memory
ManuscriptRepository rather than a real-SQLite adapter for every one of its
6 record types (ReviewerReport, ReviewComment, RevisionPlan,
RevisionProposal, RevisionVerification, ResponseToReviewers) - several of
this graph's nodes mutate that repo via direct dict access
(repo.comments[x], repo.revision_plans[id] = ...), a much tighter coupling
than write_next_section_node's single put_block() call. LangGraph's
checkpointer already durably persists the full ReviewState (including all
of the above) across all 3 human gates, the same mechanism already trusted
for the assessment and drafting gates. One lightweight summary record,
PeerReviewRound, is persisted separately (from the CLI, not a graph node)
once a round reaches a terminal state, for cross-run visibility - mirrors
the ReadinessReport/CompletionPlan "parent summary record" pattern rather
than adding 6 new tables for state the checkpointer already owns.
"""

REVIEW_CHOICES = (
    "APPROVE_REVISION_PLAN", "EDIT_REVISION_PLAN", "REJECT_REVISION_PLAN",
    "APPROVE_REVISIONS", "REQUEST_MORE_REVISION", "REJECT_REVISIONS",
    "APPROVE_RESPONSE", "REQUEST_RESPONSE_REVISION", "BLOCK_RELEASE",
)


def _to_ported_block(block: ManuscriptBlock) -> PortedManuscriptBlock:
    return PortedManuscriptBlock(
        block_id=block.block_id, section_id=block.section_id, text=block.text, claim_ids=block.claim_ids,
        evidence_ids=block.evidence_ids, literature_ids=block.literature_ids,
        authoring_agent=block.authoring_agent, prompt_hash=block.prompt_hash, model_id=block.model_id,
        code_revision=block.code_revision, status=block.status, semantic_warnings=block.semantic_warnings,
    )


def _to_ported_journal_profile(journal_id: str | None, profile: dict[str, Any] | None) -> JournalProfile | None:
    if not journal_id or not profile:
        return None
    return JournalProfile(
        journal_id=journal_id, version=str(profile.get("profile_version", "")),
        required_sections=list(profile.get("manuscript_structure", {}).get("required_sections", [])),
        citation_style=str(profile.get("citation", {}).get("style_id", "")),
    )


def initial_state(
    *, project_id: str, blocks: list[ManuscriptBlock], claims: list[Claim], evidence: list[Evidence],
    journal_profile: JournalProfile | None,
) -> ReviewState:
    return {
        "project_id": project_id,
        "journal_profile": journal_profile.model_dump(mode="json") if journal_profile else {},
        "blocks": [_to_ported_block(b).model_dump(mode="json") for b in blocks],
        "claims": [_to_ported_claim(c).model_dump(mode="json") for c in claims],
        "evidence": [_to_ported_evidence(e).model_dump(mode="json") for e in evidence],
        "literature": [],
        "reviewer_personas": list(PERSONAS),
        "reviewer_index": 0,
        "review_reports": [],
        "comment_ids": [],
        "comments": [],
        "review_round": 1,
        "terminal_status": "RUNNING",
    }


def build_manuscript_review_graph(checkpoint_db_path: str | None = None):
    review_llm = StructuredLLM(os.getenv("REVIEW_MODEL", "gpt-5-mini"))
    revision_llm = StructuredLLM(os.getenv("REVISION_MODEL", "gpt-5-mini"))
    repo = ManuscriptRepository()
    return build_review_graph(review_llm, revision_llm, repo, checkpoint_db_path)
