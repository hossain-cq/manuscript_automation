from __future__ import annotations

"""LangGraph implementation for controlled manuscript drafting and peer review.

The module uses evidence-linked manuscript blocks rather than one mutable draft
string. LLMs propose text, reviews, and revisions. Deterministic validators and
human approval gates control promotion to an approved manuscript release.
"""

import difflib
import hashlib
import json
import os
import re
import uuid
from typing import Annotated, Any, Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from ...persistence.checkpointer import build_checkpointer
from ...tools.model_gateway import (
    default_extra_body,
    default_max_completion_tokens,
    get_openai_client,
    strict_json_schema,
)


# ---------------------------------------------------------------------------
# Utilities and strict schemas
# ---------------------------------------------------------------------------


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def numeric_tokens(text: str) -> set[str]:
    return set(re.findall(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?%?", text))


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Claim(StrictModel):
    claim_id: str
    text: str
    claim_type: Literal["RESULT", "METHOD", "INTERPRETATION", "HYPOTHESIS", "LITERATURE_CLAIM"]
    importance: Literal["CENTRAL", "SUPPORTING", "CONTEXTUAL"]
    evidence_ids: list[str]
    literature_ids: list[str] = Field(default_factory=list)
    approved: bool


class Evidence(StrictModel):
    evidence_id: str
    source_artifact_id: str
    evidence_type: Literal["RESULT", "FIGURE", "TABLE", "CODE", "LITERATURE", "USER_INPUT"]
    location: str
    excerpt_or_value: str
    units: str | None = None


class JournalProfile(StrictModel):
    journal_id: str
    version: str
    required_sections: list[str]
    abstract_word_limit: int | None = None
    manuscript_word_limit: int | None = None
    required_statements: list[str] = Field(default_factory=list)
    citation_style: str


class SectionSpec(StrictModel):
    section_id: str
    title: str
    section_type: Literal[
        "TITLE", "ABSTRACT", "INTRODUCTION", "METHODS", "RESULTS",
        "DISCUSSION", "CONCLUSION", "DATA_AVAILABILITY", "CODE_AVAILABILITY",
        "AUTHOR_CONTRIBUTIONS", "CONFLICTS", "ACKNOWLEDGEMENTS", "REFERENCES"
    ]
    required_claim_ids: list[str]
    required_evidence_ids: list[str]
    word_limit: int | None = None
    generation_order: int
    approved: bool


class ManuscriptBlock(StrictModel):
    block_id: str
    section_id: str
    text: str
    claim_ids: list[str]
    evidence_ids: list[str]
    literature_ids: list[str]
    source_block_ids: list[str] = Field(default_factory=list)
    authoring_agent: str
    prompt_hash: str
    model_id: str
    code_revision: str
    status: Literal["PROPOSED", "REVIEWED", "APPROVED", "REJECTED", "SUPERSEDED"]
    semantic_warnings: list[str] = Field(default_factory=list)


class DraftBlockProposal(StrictModel):
    section_id: str
    text: str
    claim_ids: list[str]
    evidence_ids: list[str]
    literature_ids: list[str]
    warnings: list[str]
    abstain: bool


class ReviewComment(StrictModel):
    comment_id: str
    reviewer_id: str
    severity: Literal["MAJOR", "MINOR", "TECHNICAL", "EDITORIAL"]
    category: Literal[
        "SCIENTIFIC_VALIDITY", "METHODS", "EVIDENCE", "STATISTICS", "REPRODUCIBILITY",
        "LITERATURE", "OVERCLAIM", "FIGURE_TABLE", "JOURNAL_FIT", "LANGUAGE", "OTHER"
    ]
    text: str
    affected_section_ids: list[str]
    affected_block_ids: list[str]
    affected_claim_ids: list[str]
    evidence_ids: list[str]
    validity: Literal["VALID", "PARTIALLY_VALID", "UNCLEAR", "INVALID"]
    status: Literal["OPEN", "UNDER_REVIEW", "ADDRESSED", "PARTIALLY_ADDRESSED", "REJECTED_WITH_JUSTIFICATION"]
    rationale: str | None = None


class ReviewerReport(StrictModel):
    reviewer_id: str
    persona: Literal[
        "SCIENTIFIC_EXPERT", "CRITICAL_REVIEWER", "STATISTICAL_COMPUTATIONAL",
        "JOURNAL_REVIEWER", "HOSTILE_REVIEWER"
    ]
    summary: str
    comments: list[ReviewComment]
    overall_recommendation: Literal["FAVORABLE", "MINOR_REVISION", "MAJOR_REVISION", "NOT_READY"]
    evidence_ids: list[str]
    abstain: bool


class RevisionProposal(StrictModel):
    comment_id: str
    section_id: str
    original_block_id: str
    revised_text: str
    retained_claim_ids: list[str]
    added_claim_ids: list[str]
    removed_claim_ids: list[str]
    evidence_ids: list[str]
    change_summary: str
    response_to_reviewer: str
    requires_human_review: bool
    abstain: bool


class RevisionPlan(StrictModel):
    plan_id: str
    comment_ids: list[str]
    accepted_comment_ids: list[str]
    rejected_comment_ids: list[str]
    revision_tasks: list[str]
    rationale: str


class RevisionVerification(StrictModel):
    comment_id: str
    status: Literal["PASS", "WARNING", "FAIL"]
    findings: list[str]
    new_numeric_tokens: list[str]
    unsupported_claim_ids: list[str]
    evidence_ids: list[str]
    requires_human_review: bool


class ResponseToReviewers(StrictModel):
    round_id: str
    responses: list[dict[str, Any]]
    unresolved_comment_ids: list[str]
    human_approval_required: bool


# ---------------------------------------------------------------------------
# LLM gateway
# ---------------------------------------------------------------------------

class StructuredLLM:
    def __init__(self, model: str = "gpt-5-mini") -> None:
        self.client = get_openai_client()
        self.model = model

    def call(self, system: str, payload: dict[str, Any], schema: type[BaseModel]) -> BaseModel:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": schema.__name__.lower(),
                    "strict": True,
                    "schema": strict_json_schema(schema),
                },
            },
            max_completion_tokens=default_max_completion_tokens(8000),
            extra_body=default_extra_body(),
        )
        content = response.choices[0].message.content
        if not content:
            raise ValueError("Structured LLM response was empty")
        return schema.model_validate_json(content)


# ---------------------------------------------------------------------------
# Repository
# ---------------------------------------------------------------------------

class ManuscriptRepository:
    def __init__(self) -> None:
        self.blocks: dict[str, ManuscriptBlock] = {}
        self.reviews: dict[str, ReviewerReport] = {}
        self.comments: dict[str, ReviewComment] = {}
        self.revision_plans: dict[str, RevisionPlan] = {}
        self.verifications: dict[str, RevisionVerification] = {}
        self.responses: dict[str, ResponseToReviewers] = {}

    def put_block(self, block: ManuscriptBlock) -> str:
        self.blocks[block.block_id] = block
        return block.block_id

    def put_review(self, report: ReviewerReport) -> str:
        self.reviews[report.reviewer_id] = report
        for comment in report.comments:
            self.comments[comment.comment_id] = comment
        return report.reviewer_id


# ---------------------------------------------------------------------------
# Prompt policies
# ---------------------------------------------------------------------------

WRITER_SYSTEM = """You are a conservative scientific manuscript section writer.
Write only from the supplied approved claims and evidence. Never invent numbers,
experiments, methods, references, software versions, or conclusions. Preserve
the scientific scope of each claim. Every material statement must be linked to
claim IDs and evidence IDs. If evidence is missing, abstain and return warnings.
Do not silently strengthen an interpretation into a fact."""

REVIEWER_SYSTEM = """You are simulating a scientific peer reviewer.
Review only the supplied manuscript blocks, claims, evidence, literature, and
journal profile. Do not invent reviewer concerns unrelated to the supplied
content. Every substantive criticism must identify an affected block or claim
and cite supplied evidence IDs when possible. Distinguish valid concerns from
unclear or invalid requests. Report at most 3 comments - the most important
ones only - and keep each comment's text and rationale under 40 words each.
The full response must fit in a short reply; do not pad or repeat yourself."""

REVISION_SYSTEM = """You are a conservative scientific revision agent.
Revise only the affected manuscript block in response to the supplied review
comment. Do not add new numerical values, experiments, citations, methods, or
scientific conclusions unless they already exist in the supplied approved
claims/evidence. Preserve supported content. If the comment requires new work,
return abstain=true and propose a research-completion task instead."""

RESPONSE_SYSTEM = """You are preparing a response-to-reviewers document.
Use only the supplied reviewer comments, revision records, evidence, and human
decisions. Do not claim that a change was made unless a revised block exists.
For rejected comments, provide a respectful scientific justification grounded in
the supplied evidence. Mark unresolved comments explicitly."""


# ---------------------------------------------------------------------------
# Drafting graph state and nodes
# ---------------------------------------------------------------------------

class DraftState(TypedDict, total=False):
    project_id: str
    journal_profile: dict[str, Any]
    section_specs: list[dict[str, Any]]
    current_section_index: int
    claims: list[dict[str, Any]]
    evidence: list[dict[str, Any]]
    literature: list[dict[str, Any]]
    blocks: Annotated[list[dict[str, Any]], list.__add__]
    block_ids: Annotated[list[str], list.__add__]
    draft_status: Literal["RUNNING", "WAITING_FOR_HUMAN", "SUCCEEDED", "BLOCKED"]
    human_decision: dict[str, Any]
    error: str | None


def _section_context(state: DraftState, spec: SectionSpec) -> dict[str, Any]:
    claims = [
        Claim.model_validate(x) for x in state["claims"]
        if x["claim_id"] in set(spec.required_claim_ids)
    ]
    evidence = [
        Evidence.model_validate(x) for x in state["evidence"]
        if x["evidence_id"] in set(spec.required_evidence_ids)
    ]
    literature = [x for x in state.get("literature", []) if x.get("literature_id") in {i for i in spec.required_evidence_ids}]
    return {
        "section_spec": spec.model_dump(mode="json"),
        "claims": [x.model_dump(mode="json") for x in claims],
        "evidence": [x.model_dump(mode="json") for x in evidence],
        "literature": literature,
        "existing_blocks": state.get("blocks", []),
    }


def write_next_section_node(state: DraftState, llm: StructuredLLM, repo: ManuscriptRepository) -> dict[str, Any]:
    specs = [SectionSpec.model_validate(x) for x in state["section_specs"]]
    index = state.get("current_section_index", 0)
    if index >= len(specs):
        return {"draft_status": "SUCCEEDED"}
    spec = specs[index]
    context = _section_context(state, spec)
    proposal = llm.call(WRITER_SYSTEM, context, DraftBlockProposal)
    if proposal.section_id != spec.section_id:
        raise ValueError("Writer returned a different section ID")
    allowed_claims = set(spec.required_claim_ids)
    allowed_evidence = set(spec.required_evidence_ids)
    if not set(proposal.claim_ids).issubset(allowed_claims):
        raise ValueError("Writer referenced a claim not approved for this section")
    if not set(proposal.evidence_ids).issubset(allowed_evidence):
        raise ValueError("Writer referenced evidence not approved for this section")
    if proposal.abstain:
        return {
            "draft_status": "BLOCKED",
            "error": f"Section {spec.section_id} lacks sufficient approved evidence: {proposal.warnings}",
        }
    block = ManuscriptBlock(
        block_id=new_id("BLOCK"),
        section_id=spec.section_id,
        text=proposal.text,
        claim_ids=proposal.claim_ids,
        evidence_ids=proposal.evidence_ids,
        literature_ids=proposal.literature_ids,
        authoring_agent=f"writer.{spec.section_type.lower()}",
        prompt_hash=sha256_json(context),
        model_id=llm.model,
        code_revision=os.getenv("CODE_REVISION", "unknown"),
        status="PROPOSED",
        semantic_warnings=proposal.warnings,
    )
    repo.put_block(block)
    return {
        "blocks": [block.model_dump(mode="json")],
        "block_ids": [block.block_id],
        "current_section_index": index + 1,
        "draft_status": "RUNNING",
    }


def route_next_section(state: DraftState) -> str:
    if state.get("draft_status") == "BLOCKED":
        return "blocked"
    if state.get("current_section_index", 0) >= len(state["section_specs"]):
        return "review"
    return "next"


def human_draft_review_node(state: DraftState) -> dict[str, Any]:
    decision = interrupt({
        "kind": "DRAFT_REVIEW",
        "project_id": state["project_id"],
        "block_ids": state.get("block_ids", []),
        "choices": ["APPROVE_DRAFT_FOR_PEER_REVIEW", "REQUEST_DRAFT_REVISION", "BLOCK_DRAFT"],
    })
    value = decision if isinstance(decision, dict) else {"decision": decision}
    if value.get("decision") == "BLOCK_DRAFT":
        return {"draft_status": "BLOCKED", "human_decision": value}
    return {"draft_status": "SUCCEEDED", "human_decision": value}


def build_drafting_graph(llm: StructuredLLM, repo: ManuscriptRepository, checkpoint_db_path: str | None = None):
    builder = StateGraph(DraftState)
    builder.add_node("write_next_section", lambda s: write_next_section_node(s, llm, repo))
    builder.add_node("human_draft_review", human_draft_review_node)
    builder.add_node("blocked", lambda s: {"draft_status": "BLOCKED"})
    builder.add_edge(START, "write_next_section")
    builder.add_conditional_edges(
        "write_next_section",
        route_next_section,
        {"next": "write_next_section", "review": "human_draft_review", "blocked": "blocked"},
    )
    builder.add_edge("human_draft_review", END)
    builder.add_edge("blocked", END)
    if checkpoint_db_path is None:
        return builder.compile()
    return builder.compile(checkpointer=build_checkpointer(checkpoint_db_path))


# ---------------------------------------------------------------------------
# Peer-review simulation and revision graph state
# ---------------------------------------------------------------------------

class ReviewState(TypedDict, total=False):
    project_id: str
    journal_profile: dict[str, Any]
    blocks: list[dict[str, Any]]
    claims: list[dict[str, Any]]
    evidence: list[dict[str, Any]]
    literature: list[dict[str, Any]]
    reviewer_personas: list[str]
    reviewer_index: int
    review_reports: Annotated[list[dict[str, Any]], list.__add__]
    comment_ids: Annotated[list[str], list.__add__]
    comments: list[dict[str, Any]]
    revision_plan: dict[str, Any]
    revision_proposals: Annotated[list[dict[str, Any]], list.__add__]
    revision_verifications: Annotated[list[dict[str, Any]], list.__add__]
    response_to_reviewers: dict[str, Any]
    review_round: int
    human_decision: dict[str, Any]
    terminal_status: Literal["RUNNING", "WAITING_FOR_HUMAN", "SUCCEEDED", "BLOCKED"]
    error: str | None


PERSONAS = [
    "SCIENTIFIC_EXPERT",
    "CRITICAL_REVIEWER",
    "STATISTICAL_COMPUTATIONAL",
    "JOURNAL_REVIEWER",
    "HOSTILE_REVIEWER",
]

PERSONA_INSTRUCTIONS = {
    "SCIENTIFIC_EXPERT": "Focus on novelty, scientific validity, methodology, and interpretation.",
    "CRITICAL_REVIEWER": "Look for unsupported claims, missing controls, weak comparisons, and overclaiming.",
    "STATISTICAL_COMPUTATIONAL": "Check data splits, uncertainty, convergence, statistics, benchmarking, and reproducibility.",
    "JOURNAL_REVIEWER": "Check scope, structure, significance, article type, clarity, and journal fit.",
    "HOSTILE_REVIEWER": "Actively search for serious weaknesses, but do not invent evidence or demands unrelated to the manuscript.",
}


def reviewer_node(state: ReviewState, llm: StructuredLLM, repo: ManuscriptRepository) -> dict[str, Any]:
    index = state.get("reviewer_index", 0)
    personas = state.get("reviewer_personas", PERSONAS)
    if index >= len(personas):
        return {"terminal_status": "RUNNING"}
    persona = personas[index]
    report = llm.call(
        REVIEWER_SYSTEM,
        {
            "reviewer_id": f"reviewer-{persona.lower()}",
            "persona": persona,
            "persona_instruction": PERSONA_INSTRUCTIONS[persona],
            "journal_profile": state["journal_profile"],
            "blocks": state["blocks"],
            "claims": state["claims"],
            "evidence": state["evidence"],
            "literature": state.get("literature", []),
        },
        ReviewerReport,
    )
    if report.reviewer_id != f"reviewer-{persona.lower()}":
        raise ValueError("Reviewer returned an unexpected reviewer ID")
    # Each persona is called independently and tends to default to the same
    # short comment_id scheme ("COMMENT-1", "COMMENT-2", ...) - confirmed
    # against a real 5-persona run. ManuscriptRepository.comments is a flat
    # dict keyed by comment_id, so without namespacing, a later persona's
    # "COMMENT-1" silently overwrites an earlier persona's "COMMENT-1",
    # losing that reviewer's comment entirely. Prefix by persona so ids are
    # globally unique across the whole review round.
    report = report.model_copy(update={
        "comments": [c.model_copy(update={"comment_id": f"{persona}-{c.comment_id}"}) for c in report.comments],
    })
    repo.put_review(report)
    return {
        "review_reports": [report.model_dump(mode="json")],
        "comment_ids": [x.comment_id for x in report.comments],
        "reviewer_index": index + 1,
    }


def route_reviewers(state: ReviewState) -> str:
    return "next" if state.get("reviewer_index", 0) < len(state.get("reviewer_personas", PERSONAS)) else "triage"


def triage_comments_node(state: ReviewState, llm: StructuredLLM, repo: ManuscriptRepository) -> dict[str, Any]:
    comments = [repo.comments[x] for x in state.get("comment_ids", []) if x in repo.comments]
    accepted = [c.comment_id for c in comments if c.validity in {"VALID", "PARTIALLY_VALID", "UNCLEAR"}]
    rejected = [c.comment_id for c in comments if c.validity == "INVALID"]
    tasks = [
        f"Address {comment.comment_id}: {comment.text}"
        for comment in comments
        if comment.comment_id in accepted
    ]
    plan = RevisionPlan(
        plan_id=new_id("REVPLAN"),
        comment_ids=[c.comment_id for c in comments],
        accepted_comment_ids=accepted,
        rejected_comment_ids=rejected,
        revision_tasks=tasks,
        rationale="Comments are triaged by validity and affected artifacts; unclear comments remain human-reviewable.",
    )
    repo.revision_plans[plan.plan_id] = plan
    return {"revision_plan": plan.model_dump(mode="json"), "terminal_status": "WAITING_FOR_HUMAN"}


def human_revision_plan_review_node(state: ReviewState) -> dict[str, Any]:
    decision = interrupt({
        "kind": "REVISION_PLAN_REVIEW",
        "round": state.get("review_round", 1),
        "revision_plan": state["revision_plan"],
        "review_reports": state.get("review_reports", []),
        "choices": ["APPROVE_REVISION_PLAN", "EDIT_REVISION_PLAN", "REJECT_REVISION_PLAN"],
    })
    value = decision if isinstance(decision, dict) else {"decision": decision}
    if value.get("decision") == "REJECT_REVISION_PLAN":
        return {"terminal_status": "BLOCKED", "human_decision": value}
    return {"terminal_status": "RUNNING", "human_decision": value}


def revise_comment_node(
    state: ReviewState,
    llm: StructuredLLM,
    repo: ManuscriptRepository,
) -> dict[str, Any]:
    comments = [repo.comments[x] for x in state["revision_plan"]["accepted_comment_ids"] if x in repo.comments]
    proposals: list[dict[str, Any]] = []
    for comment in comments:
        affected = [
            ManuscriptBlock.model_validate(x)
            for x in state["blocks"]
            if x["block_id"] in set(comment.affected_block_ids)
        ]
        if not affected:
            continue
        response = llm.call(
            REVISION_SYSTEM,
            {
                "comment": comment.model_dump(mode="json"),
                "affected_blocks": [x.model_dump(mode="json") for x in affected],
                "approved_claims": state["claims"],
                "approved_evidence": state["evidence"],
                "literature": state.get("literature", []),
            },
            RevisionProposal,
        )
        if response.comment_id != comment.comment_id:
            raise ValueError("Revision proposal returned unexpected comment ID")
        if response.abstain:
            proposals.append(response.model_dump(mode="json"))
            continue
        allowed_claims = {x["claim_id"] for x in state["claims"] if x.get("approved")}
        allowed_evidence = {x["evidence_id"] for x in state["evidence"]}
        if not set(response.retained_claim_ids + response.added_claim_ids).issubset(allowed_claims):
            raise ValueError("Revision attempted to add an unapproved claim")
        if not set(response.evidence_ids).issubset(allowed_evidence):
            raise ValueError("Revision referenced evidence outside the approved evidence set")
        proposals.append(response.model_dump(mode="json"))
    return {"revision_proposals": proposals}


def verify_revisions_node(state: ReviewState, repo: ManuscriptRepository) -> dict[str, Any]:
    verifications: list[dict[str, Any]] = []
    current_by_id = {x["block_id"]: x for x in state["blocks"]}
    approved_evidence = {x["evidence_id"] for x in state["evidence"]}
    for proposal_raw in state.get("revision_proposals", []):
        proposal = RevisionProposal.model_validate(proposal_raw)
        original = current_by_id.get(proposal.original_block_id, {})
        old_numbers = numeric_tokens(original.get("text", ""))
        new_numbers = numeric_tokens(proposal.revised_text)
        added_numbers = sorted(new_numbers - old_numbers)
        unsupported = [
            x for x in proposal.evidence_ids if x not in approved_evidence
        ]
        if proposal.abstain:
            status, findings = "WARNING", ["Revision agent abstained; research completion may be required."]
        elif unsupported:
            status, findings = "FAIL", ["Revision references evidence outside approved evidence set."]
        elif added_numbers:
            status, findings = "WARNING", ["Revision introduces new numeric tokens; human verification required."]
        else:
            status, findings = "PASS", []
        verification = RevisionVerification(
            comment_id=proposal.comment_id,
            status=status,
            findings=findings,
            new_numeric_tokens=added_numbers,
            unsupported_claim_ids=unsupported,
            evidence_ids=proposal.evidence_ids,
            requires_human_review=status != "PASS" or proposal.requires_human_review,
        )
        repo.verifications[proposal.comment_id] = verification
        verifications.append(verification.model_dump(mode="json"))
    return {"revision_verifications": verifications}


def human_revision_review_node(state: ReviewState) -> dict[str, Any]:
    decision = interrupt({
        "kind": "REVISION_REVIEW",
        "round": state.get("review_round", 1),
        "revision_proposals": state.get("revision_proposals", []),
        "revision_verifications": state.get("revision_verifications", []),
        "choices": ["APPROVE_REVISIONS", "REQUEST_MORE_REVISION", "REJECT_REVISIONS"],
    })
    value = decision if isinstance(decision, dict) else {"decision": decision}
    if value.get("decision") == "REJECT_REVISIONS":
        return {"terminal_status": "BLOCKED", "human_decision": value}
    return {"terminal_status": "RUNNING", "human_decision": value}


def response_to_reviewers_node(state: ReviewState, llm: StructuredLLM, repo: ManuscriptRepository) -> dict[str, Any]:
    responses: list[dict[str, Any]] = []
    unresolved: list[str] = []
    for comment_id in state["revision_plan"]["comment_ids"]:
        comment = repo.comments.get(comment_id)
        if not comment:
            continue
        proposal = next((x for x in state.get("revision_proposals", []) if x["comment_id"] == comment_id), None)
        verification = next((x for x in state.get("revision_verifications", []) if x["comment_id"] == comment_id), None)
        if proposal is None or (verification and verification["status"] == "FAIL"):
            unresolved.append(comment_id)
            responses.append({"comment_id": comment_id, "status": "UNRESOLVED", "response": "This comment remains unresolved and requires author action."})
        elif proposal.get("abstain"):
            unresolved.append(comment_id)
            responses.append({"comment_id": comment_id, "status": "RESEARCH_REQUIRED", "response": proposal.get("change_summary", "Additional work is required.")})
        else:
            responses.append({
                "comment_id": comment_id,
                "status": "ADDRESSED",
                "response": proposal["response_to_reviewer"],
                "change_summary": proposal["change_summary"],
            })
    result = ResponseToReviewers(
        round_id=f"ROUND-{state.get('review_round', 1)}",
        responses=responses,
        unresolved_comment_ids=unresolved,
        human_approval_required=True,
    )
    response_id = new_id("RESPONSE")
    repo.responses[response_id] = result
    return {"response_to_reviewers": result.model_dump(mode="json"), "terminal_status": "WAITING_FOR_HUMAN"}


def final_response_approval_node(state: ReviewState) -> dict[str, Any]:
    decision = interrupt({
        "kind": "RESPONSE_TO_REVIEWERS_APPROVAL",
        "response_to_reviewers": state["response_to_reviewers"],
        "choices": ["APPROVE_RESPONSE", "REQUEST_RESPONSE_REVISION", "BLOCK_RELEASE"],
    })
    value = decision if isinstance(decision, dict) else {"decision": decision}
    if value.get("decision") == "APPROVE_RESPONSE":
        return {"terminal_status": "SUCCEEDED", "human_decision": value}
    if value.get("decision") == "BLOCK_RELEASE":
        return {"terminal_status": "BLOCKED", "human_decision": value}
    return {"terminal_status": "RUNNING", "human_decision": value}


def build_review_graph(
    review_llm: StructuredLLM,
    revision_llm: StructuredLLM,
    repo: ManuscriptRepository,
    checkpoint_db_path: str | None = None,
):
    builder = StateGraph(ReviewState)
    builder.add_node("reviewer", lambda s: reviewer_node(s, review_llm, repo))
    builder.add_node("triage", lambda s: triage_comments_node(s, review_llm, repo))
    builder.add_node("human_plan_review", human_revision_plan_review_node)
    builder.add_node("revise_comments", lambda s: revise_comment_node(s, revision_llm, repo))
    builder.add_node("verify_revisions", lambda s: verify_revisions_node(s, repo))
    builder.add_node("human_revision_review", human_revision_review_node)
    builder.add_node("response_to_reviewers", lambda s: response_to_reviewers_node(s, revision_llm, repo))
    builder.add_node("final_response_approval", final_response_approval_node)
    builder.add_node("blocked", lambda s: {"terminal_status": "BLOCKED"})

    builder.add_edge(START, "reviewer")
    builder.add_conditional_edges(
        "reviewer",
        route_reviewers,
        {"next": "reviewer", "triage": "triage"},
    )
    builder.add_edge("triage", "human_plan_review")
    builder.add_conditional_edges(
        "human_plan_review",
        lambda s: "blocked" if s.get("terminal_status") == "BLOCKED" else "revise",
        {"revise": "revise_comments", "blocked": "blocked"},
    )
    builder.add_edge("revise_comments", "verify_revisions")
    builder.add_edge("verify_revisions", "human_revision_review")
    builder.add_conditional_edges(
        "human_revision_review",
        lambda s: "blocked" if s.get("terminal_status") == "BLOCKED" else "response",
        {"response": "response_to_reviewers", "blocked": "blocked"},
    )
    builder.add_edge("response_to_reviewers", "final_response_approval")
    builder.add_conditional_edges(
        "final_response_approval",
        lambda s: "end" if s.get("terminal_status") == "SUCCEEDED" else "blocked" if s.get("terminal_status") == "BLOCKED" else "response",
        {"end": END, "blocked": "blocked", "response": "response_to_reviewers"},
    )
    builder.add_edge("blocked", END)
    if checkpoint_db_path is None:
        return builder.compile()
    return builder.compile(checkpointer=build_checkpointer(checkpoint_db_path))


# ---------------------------------------------------------------------------
# Construction helper
# ---------------------------------------------------------------------------


def build_graphs():
    repo = ManuscriptRepository()
    review_llm = StructuredLLM(os.getenv("REVIEW_MODEL", "gpt-5"))
    revision_llm = StructuredLLM(os.getenv("REVISION_MODEL", "gpt-5-mini"))
    return build_drafting_graph(revision_llm, repo), build_review_graph(review_llm, revision_llm, repo), repo


if __name__ == "__main__":
    print("Module loaded. Use build_graphs() to compile drafting and review graphs.")
