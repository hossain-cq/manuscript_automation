from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .enums import (
    CitationVerificationStatus,
    ClaimImportance,
    ClaimType,
    CompletionTaskCategory,
    CompletionTaskPriority,
    EvidenceType,
    FindingSeverity,
    NumericCrossCheckStatus,
    ReadinessStatus,
    RunStatus,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Project(StrictModel):
    project_id: str
    source_path: str
    domain_profile_id: str | None = None
    created_at: str = Field(default_factory=utc_now)


class SourceAsset(StrictModel):
    artifact_id: str
    project_id: str
    relative_path: str
    checksum_sha256: str
    size_bytes: int
    media_type: str
    modified_at: str | None = None


class ProjectKnowledgeMap(StrictModel):
    map_id: str
    project_id: str
    run_id: str
    asset_counts_by_role: dict[str, int] = Field(default_factory=dict)
    total_assets: int = 0
    total_size_bytes: int = 0
    has_readme: bool = False
    has_environment_file: bool = False
    created_at: str = Field(default_factory=utc_now)


class Run(StrictModel):
    run_id: str
    project_id: str
    thread_id: str
    workflow_name: str
    status: RunStatus
    current_stage: str
    created_at: str = Field(default_factory=utc_now)
    updated_at: str = Field(default_factory=utc_now)


class Claim(StrictModel):
    claim_id: str
    project_id: str
    text: str
    claim_type: ClaimType
    importance: ClaimImportance = "SUPPORTING"
    evidence_ids: list[str] = Field(default_factory=list)
    status: str = "CANDIDATE"


class Evidence(StrictModel):
    evidence_id: str
    project_id: str
    source_artifact_id: str
    evidence_type: EvidenceType
    location: str
    excerpt_or_value: str
    extraction_confidence: float = Field(ge=0.0, le=1.0, default=1.0)


class Finding(StrictModel):
    finding_id: str
    project_id: str
    severity: FindingSeverity
    rule_id: str
    message: str
    blocking: bool = False
    affected_claim_ids: list[str] = Field(default_factory=list)


class HumanDecision(StrictModel):
    decision_id: str
    project_id: str
    run_id: str
    kind: str
    decision: str
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=utc_now)


class Citation(StrictModel):
    citation_id: str
    project_id: str
    cite_key: str
    bib_title: str = ""
    bib_authors: str = ""
    bib_year: int | None = None
    bib_doi: str | None = None
    verification_status: CitationVerificationStatus
    verified_title: str | None = None
    verified_doi: str | None = None
    match_confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    notes: str = ""


class ClaimAssessmentRecord(StrictModel):
    """One assessor's judgment of one claim - the reasoning behind a score,
    not just the score. Previously computed by run_audits' LLM assessors and
    discarded after being folded into the aggregate readiness score; nothing
    kept the *why*."""
    record_id: str
    project_id: str
    run_id: str
    assessor_id: str
    claim_id: str
    label: str
    score: float = Field(ge=0.0, le=1.0, default=0.0)
    reasoning: str = ""
    limitations: list[str] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)


class AssessorReport(StrictModel):
    """One assessor's full readiness judgment for a run - the per-dimension
    scores plus the qualitative signal (major_risks, contribution_candidates)
    that determine_readiness() doesn't use but a human reviewing the run
    should still be able to see."""
    report_id: str
    project_id: str
    run_id: str
    assessor_id: str
    scientific_contribution: float = Field(ge=0.0, le=1.0, default=0.0)
    evidence_sufficiency: float = Field(ge=0.0, le=1.0, default=0.0)
    methodological_rigor: float = Field(ge=0.0, le=1.0, default=0.0)
    validation_strength: float = Field(ge=0.0, le=1.0, default=0.0)
    reproducibility: float = Field(ge=0.0, le=1.0, default=0.0)
    literature_positioning: float = Field(ge=0.0, le=1.0, default=0.0)
    potential_significance: float = Field(ge=0.0, le=1.0, default=0.0)
    major_risks: list[str] = Field(default_factory=list)
    contribution_candidates: list[dict[str, Any]] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    abstain: bool = False
    created_at: str = Field(default_factory=utc_now)


class NumericCrossCheck(StrictModel):
    check_id: str
    project_id: str
    claimed_value: float
    claimed_text: str
    source_section: str
    status: NumericCrossCheckStatus
    matched_value: float | None = None
    matched_source_path: str | None = None
    matched_context: str | None = None
    precision_places: int = 0


class CompletionTask(StrictModel):
    task_id: str
    project_id: str
    plan_id: str
    title: str
    reason: str
    category: CompletionTaskCategory
    priority: CompletionTaskPriority
    affected_claim_ids: list[str] = Field(default_factory=list)
    source: str = ""


class CompletionPlan(StrictModel):
    plan_id: str
    project_id: str
    run_id: str
    task_ids: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=utc_now)


class PlannedSection(StrictModel):
    section_id: str
    project_id: str
    plan_id: str
    name: str
    order: int
    claim_ids: list[str] = Field(default_factory=list)


class ManuscriptPlan(StrictModel):
    plan_id: str
    project_id: str
    run_id: str
    journal_id: str | None = None
    section_ids: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=utc_now)


class ManuscriptBlock(StrictModel):
    block_id: str
    project_id: str
    plan_id: str
    section_id: str
    text: str
    claim_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    literature_ids: list[str] = Field(default_factory=list)
    authoring_agent: str = ""
    prompt_hash: str = ""
    model_id: str = ""
    code_revision: str = ""
    status: str = "PROPOSED"
    semantic_warnings: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=utc_now)


class PeerReviewRound(StrictModel):
    round_id: str
    project_id: str
    plan_id: str
    thread_id: str
    status: str
    response_to_reviewers: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=utc_now)


class ReleaseCandidate(StrictModel):
    release_id: str
    project_id: str
    manuscript_plan_id: str | None = None
    readiness_report_id: str | None = None
    completion_plan_id: str | None = None
    peer_review_round_id: str | None = None
    section_count: int = 0
    drafted_section_count: int = 0
    citation_count: int = 0
    human_decision_count: int = 0
    package_hash: str = ""
    created_at: str = Field(default_factory=utc_now)


class ReadinessReport(StrictModel):
    report_id: str
    project_id: str
    run_id: str
    readiness_status: ReadinessStatus
    audit_ids: list[str] = Field(default_factory=list)
    finding_ids: list[str] = Field(default_factory=list)
    blocking_finding_ids: list[str] = Field(default_factory=list)
    explanation: str = ""
    created_at: str = Field(default_factory=utc_now)
