from __future__ import annotations

import operator
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from ..domain.enums import ReadinessStatus, RunStatus
from ..domain.models import (
    AssessorReport,
    ClaimAssessmentRecord,
    CompletionPlan,
    CompletionTask,
    Finding,
    HumanDecision,
    ManuscriptPlan,
    PlannedSection,
    ProjectKnowledgeMap,
    ReadinessReport,
    SourceAsset,
)
from ..persistence.checkpointer import build_checkpointer
from ..persistence.repositories import Repository, new_id
from ..tools.data_sufficiency import check_data_sufficiency
from ..tools.filesystem import BoundaryError, scan_project, validate_project_path
from ..tools.journals import load_journal_profile as load_target_journal_profile
from .evidence_extraction import (
    classify_domain_from_assets,
    extract_evidence_and_claims,
    load_domain_profile,
    to_evidence_packet,
)
from .subgraphs.novelty_and_publishability import (
    READINESS_SYSTEM,
    InMemoryRepository,
    LLMReadinessAssessment,
    StructuredLLMGateway,
    build_assessor_prompt,
    compare_assessments,
    compute_empirical_features,
    determine_readiness,
    evaluate_blockers,
    lexical_similarity,
)

"""Top-level project-assessment graph.

Ported from the repo-root supervisor_graph.py prototype. Same node sequence
and same routing shape; the fake persist_*/write_* helper functions are
replaced with the real Repository (SQLite), and validate_boundary/create_manifest
now do real path validation and file scanning instead of returning fake IDs.

classify_domain and run_audits are real: they call
graphs/evidence_extraction.py to turn scanned files into Claim/Evidence rows,
then (if any were extracted) run the publishability-assessment logic from
graphs/subgraphs/novelty_and_publishability.py directly - composing that
module's functions rather than invoking its compiled graph, which has its own
human_review interrupt that would otherwise double up with this graph's own
human_assessment_review gate. See docs in graphs/evidence_extraction.py.
"""

ASSESSMENT_CHOICES = (
    "APPROVE_MANUSCRIPT_PLANNING",
    "APPROVE_COMPLETION_PLAN",
    "REQUEST_REASSESSMENT",
    "BLOCK_RUN",
)


class RunContext(TypedDict, total=False):
    project_id: str
    run_id: str
    thread_id: str
    workflow_name: str
    status: RunStatus
    current_stage: str
    created_at: str
    updated_at: str


class AssessmentState(TypedDict, total=False):
    context: RunContext
    source_path: str
    target_journal_id: str | None
    source_manifest_id: str | None
    source_artifact_ids: list[str]
    knowledge_map_id: str | None
    domain_profile_id: str | None
    domain_confidence: float | None

    audit_ids: Annotated[list[str], operator.add]
    finding_ids: Annotated[list[str], operator.add]
    candidate_claim_ids: Annotated[list[str], operator.add]
    blocking_finding_ids: Annotated[list[str], operator.add]

    readiness_report_id: str | None
    readiness_status: ReadinessStatus | None
    readiness_explanation: str | None
    human_decision: dict[str, Any] | None
    completion_plan_id: str | None
    manuscript_plan_id: str | None
    error: dict[str, Any] | None


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


def initialize_run(state: AssessmentState) -> dict[str, Any]:
    context = state["context"]
    return {
        "context": {**context, "status": "RUNNING", "current_stage": "BOUNDARY_VALIDATION", "updated_at": now_iso()},
        "error": None,
    }


def validate_boundary(state: AssessmentState) -> dict[str, Any]:
    try:
        validate_project_path(state["source_path"])
    except BoundaryError as exc:
        return {
            "context": {**state["context"], "status": "BLOCKED", "updated_at": now_iso()},
            "error": {"code": "INVALID_SOURCE_PATH", "message": str(exc)},
        }
    return {"context": {**state["context"], "current_stage": "MANIFEST_CREATION", "updated_at": now_iso()}}


def create_manifest(state: AssessmentState, repo: Repository) -> dict[str, Any]:
    root = validate_project_path(state["source_path"])
    scanned = scan_project(root)
    assets = [
        SourceAsset(
            artifact_id=new_id("ASSET"),
            project_id=state["context"]["project_id"],
            relative_path=scanned_file.relative_path,
            checksum_sha256=scanned_file.checksum_sha256,
            size_bytes=scanned_file.size_bytes,
            media_type=scanned_file.media_type,
        )
        for scanned_file in scanned
    ]
    repo.add_source_assets(assets)
    return {
        "source_manifest_id": new_id("MANIFEST"),
        "source_artifact_ids": [asset.artifact_id for asset in assets],
        "context": {**state["context"], "current_stage": "DISCOVERY", "updated_at": now_iso()},
    }


_ROLE_EXTENSIONS = {
    "CODE": {".py", ".ipynb"},
    "DATA": {".csv", ".dat"},
    "FIGURE": {".jpg", ".jpeg", ".png", ".pdf"},
    "DOCUMENT": {".md", ".txt"},
    "CONFIG": {".yaml", ".yml", ".json"},
}
_ENVIRONMENT_FILENAMES = {
    "environment.yml", "environment.yaml", "requirements.txt",
    "pyproject.toml", "pipfile", "setup.py",
}


def _asset_role(relative_path: str) -> str:
    suffix = "." + relative_path.rsplit(".", 1)[-1].lower() if "." in relative_path else ""
    for role, extensions in _ROLE_EXTENSIONS.items():
        if suffix in extensions:
            return role
    return "OTHER"


def discover_project(state: AssessmentState, repo: Repository) -> dict[str, Any]:
    """Domain-agnostic project structure summary, built entirely from the
    SourceAsset rows create_manifest already persisted - no new scanning.
    Runs before classify_domain, so it can't use domain checklists (that's
    evidence_extraction.py's job, deliberately deferred until a domain
    profile is known); what it can do without duplicating that work is
    categorize files by role and flag basic reproducibility signals."""
    project_id = state["context"]["project_id"]
    run_id = state["context"]["run_id"]
    assets = repo.get_source_assets(project_id)

    counts: dict[str, int] = {}
    has_readme = False
    has_environment_file = False
    for asset in assets:
        role = _asset_role(asset.relative_path)
        counts[role] = counts.get(role, 0) + 1
        basename = asset.relative_path.rsplit("/", 1)[-1].lower()
        if basename.startswith("readme"):
            has_readme = True
        if basename in _ENVIRONMENT_FILENAMES:
            has_environment_file = True

    findings: list[Finding] = []
    if not has_readme:
        findings.append(Finding(
            finding_id=new_id("FINDING"), project_id=project_id, severity="MEDIUM",
            rule_id="missing_readme", message="No README file found at the project root.",
        ))
    if not has_environment_file:
        findings.append(Finding(
            finding_id=new_id("FINDING"), project_id=project_id, severity="MEDIUM",
            rule_id="missing_environment_file",
            message="No environment/dependency file found (environment.yml, requirements.txt, pyproject.toml, ...).",
        ))
    if not counts.get("CODE"):
        findings.append(Finding(
            finding_id=new_id("FINDING"), project_id=project_id, severity="MEDIUM",
            rule_id="no_code_files_found", message="No code files (.py, .ipynb) found in the project.",
        ))
    if not counts.get("DATA") and not counts.get("FIGURE"):
        findings.append(Finding(
            finding_id=new_id("FINDING"), project_id=project_id, severity="MEDIUM",
            rule_id="no_data_or_figures_found", message="No data files (.csv, .dat) or figures found in the project.",
        ))
    for finding in findings:
        repo.add_finding(finding)

    map_id = new_id("KMAP")
    repo.add_knowledge_map(ProjectKnowledgeMap(
        map_id=map_id, project_id=project_id, run_id=run_id, asset_counts_by_role=counts,
        total_assets=len(assets), total_size_bytes=sum(a.size_bytes for a in assets),
        has_readme=has_readme, has_environment_file=has_environment_file,
    ))

    return {
        "knowledge_map_id": map_id,
        "finding_ids": [f.finding_id for f in findings],
        "context": {**state["context"], "current_stage": "DOMAIN_CLASSIFICATION", "updated_at": now_iso()},
    }


def classify_domain(state: AssessmentState, repo: Repository) -> dict[str, Any]:
    # Deterministic (no LLM): scores every configs/domains/*.yaml profile
    # against the scanned file paths. See evidence_extraction.py.
    assets = repo.get_source_assets(state["context"]["project_id"])
    domain_id, confidence = classify_domain_from_assets(assets)
    return {
        "domain_profile_id": domain_id,
        "domain_confidence": confidence,
        "context": {**state["context"], "current_stage": "READINESS_AUDIT", "updated_at": now_iso()},
    }


def run_audits(state: AssessmentState, repo: Repository) -> dict[str, Any]:
    project_id = state["context"]["project_id"]
    domain_id = state.get("domain_profile_id") or "generic"

    assets = repo.get_source_assets(project_id)
    domain_profile = load_domain_profile(domain_id)
    claims, evidence, findings = extract_evidence_and_claims(assets, domain_profile, project_id)

    sufficiency_findings = check_data_sufficiency(assets, Path(state["source_path"]), project_id)

    repo.add_claims(claims)
    repo.add_evidence_items(evidence)
    for finding in findings + sufficiency_findings:
        repo.add_finding(finding)

    result: dict[str, Any] = {
        "audit_ids": [new_id("AUDIT"), new_id("AUDIT")],
        "finding_ids": [f.finding_id for f in findings + sufficiency_findings],
        "candidate_claim_ids": [c.claim_id for c in claims],
        "blocking_finding_ids": [],
        "context": {**state["context"], "current_stage": "READINESS_SYNTHESIS", "updated_at": now_iso()},
    }

    if not claims:
        # No evidence extracted (synthetic/empty project, or an unrecognized
        # domain) - skip the LLM rather than spend a call on nothing. Keeps
        # this path zero-LLM-dependency, same as the Phase 0 tests expect.
        return result

    packet = to_evidence_packet(
        project_id=project_id, domain_profile_id=domain_id, claims=claims, evidence=evidence,
    )
    audit_repo = InMemoryRepository()
    feature_ids = compute_empirical_features(packet, audit_repo)
    # compute_empirical_features stores what it computes in audit_repo, not
    # back onto the packet - but determine_readiness reads coverage from
    # packet.empirical_features, a *separate* field only meant for features
    # supplied before the packet was built. Every real caller constructs the
    # packet with empirical_features=[] (there's no other source at this
    # point), so left as-is, determine_readiness always sees coverage=0.0 and
    # returns INSUFFICIENT_EVIDENCE regardless of actual coverage - confirmed
    # against AQT_electrolyte, which has real evidence for every claim. Feed
    # what was just computed back onto the packet before scoring readiness.
    packet = packet.model_copy(update={
        "empirical_features": [audit_repo.features[i] for i in feature_ids if i in audit_repo.features],
    })
    blocker_ids = evaluate_blockers(packet, audit_repo)
    blocking_ids = [audit_repo.findings[i].finding_id for i in blocker_ids if audit_repo.findings[i].blocking]

    gateway = StructuredLLMGateway(model=os.getenv("ASSESSMENT_MODEL", "gpt-5-mini"))
    assessments: list[LLMReadinessAssessment] = []
    for assessor_id in ("contribution_assessor", "evidence_assessor", "critical_reviewer"):
        llm_result = gateway.call(
            system=READINESS_SYSTEM, user=build_assessor_prompt(packet, assessor_id), schema=LLMReadinessAssessment,
        )
        if not llm_result.evidence_ids and packet.claims:
            llm_result = llm_result.model_copy(update={"abstain": True})
        assessments.append(llm_result.model_copy(update={"assessor_id": assessor_id}))

    # Persist what each assessor actually said, not just the aggregate score
    # compare_assessments() below computes. Previously this reasoning -
    # major_risks, and each claim's reasoning/limitations/missing_evidence -
    # was discarded right after being folded into the aggregate, so nothing
    # recorded *why* a claim scored the way it did.
    run_id = state["context"]["run_id"]
    assessor_reports = [
        AssessorReport(
            report_id=new_id("ASSESSORREPORT"), project_id=project_id, run_id=run_id,
            assessor_id=a.assessor_id, scientific_contribution=a.scientific_contribution,
            evidence_sufficiency=a.evidence_sufficiency, methodological_rigor=a.methodological_rigor,
            validation_strength=a.validation_strength, reproducibility=a.reproducibility,
            literature_positioning=a.literature_positioning, potential_significance=a.potential_significance,
            major_risks=a.major_risks,
            contribution_candidates=[c.model_dump(mode="json") for c in a.contribution_candidates],
            confidence=a.confidence, abstain=a.abstain,
        )
        for a in assessments
    ]
    claim_assessment_records = [
        ClaimAssessmentRecord(
            record_id=new_id("CLAIMASSESS"), project_id=project_id, run_id=run_id,
            assessor_id=a.assessor_id, claim_id=ca.claim_id, label=ca.label, score=ca.score,
            reasoning=ca.reasoning, limitations=ca.limitations, missing_evidence=ca.missing_evidence,
            confidence=ca.confidence,
        )
        for a in assessments for ca in a.claim_assessments
    ]
    repo.add_assessor_reports(assessor_reports)
    repo.add_claim_assessments(claim_assessment_records)

    scores, _confidence, disagreement = compare_assessments(assessments)
    blocker_findings = [audit_repo.findings[i] for i in blocker_ids if i in audit_repo.findings]
    readiness_status, explanation = determine_readiness(packet, blocker_findings, scores, disagreement)

    result["blocking_finding_ids"] = blocking_ids
    result["readiness_status"] = readiness_status
    result["readiness_explanation"] = explanation
    return result


def synthesize_readiness(state: AssessmentState, repo: Repository) -> dict[str, Any]:
    if state.get("readiness_status"):
        # run_audits already produced a real, LLM-assessed readiness status.
        status: ReadinessStatus = state["readiness_status"]
        explanation = state.get("readiness_explanation") or ""
    else:
        status = (
            "DRAFTABLE_WITH_WARNINGS" if not state.get("blocking_finding_ids") else "NEEDS_ADDITIONAL_ANALYSIS"
        )
        explanation = (
            "No claims were extracted for this project's domain; readiness "
            "could not be assessed against evidence." if not state.get("candidate_claim_ids") else ""
        )
    report = ReadinessReport(
        report_id=new_id("READINESS"),
        project_id=state["context"]["project_id"],
        run_id=state["context"]["run_id"],
        readiness_status=status,
        audit_ids=state.get("audit_ids", []),
        finding_ids=state.get("finding_ids", []),
        blocking_finding_ids=state.get("blocking_finding_ids", []),
        explanation=explanation,
    )
    repo.add_readiness_report(report)
    return {
        "readiness_report_id": report.report_id,
        "readiness_status": status,
        "context": {
            **state["context"],
            "current_stage": "HUMAN_ASSESSMENT_REVIEW",
            "status": "WAITING_FOR_HUMAN",
            "updated_at": now_iso(),
        },
    }


def human_assessment_review(state: AssessmentState, repo: Repository) -> dict[str, Any]:
    payload = {
        "kind": "ASSESSMENT_REVIEW",
        "question": "Review the publication-readiness assessment.",
        "report_id": state["readiness_report_id"],
        "readiness_status": state["readiness_status"],
        "blocking_finding_ids": state.get("blocking_finding_ids", []),
        "choices": list(ASSESSMENT_CHOICES),
    }
    response = interrupt(payload)
    decision = response.get("decision") if isinstance(response, dict) else response
    if decision not in ASSESSMENT_CHOICES:
        raise ValueError(f"Invalid human decision: {decision!r}")

    repo.add_human_decision(HumanDecision(
        decision_id=new_id("DECISION"),
        project_id=state["context"]["project_id"],
        run_id=state["context"]["run_id"],
        kind=payload["kind"],
        decision=decision,
        payload=payload,
    ))
    return {
        "human_decision": {"decision": decision},
        "context": {
            **state["context"], "status": "RUNNING",
            "current_stage": "ROUTING_AFTER_ASSESSMENT", "updated_at": now_iso(),
        },
    }


def route_after_assessment(state: AssessmentState) -> str:
    decision = (state.get("human_decision") or {}).get("decision")
    return {
        "APPROVE_MANUSCRIPT_PLANNING": "manuscript",
        "APPROVE_COMPLETION_PLAN": "completion",
        "REQUEST_REASSESSMENT": "reassess",
    }.get(decision, "blocked")


def create_completion_plan_node(state: AssessmentState, repo: Repository) -> dict[str, Any]:
    """Turns Findings (unambiguous "no file matched this checklist item"
    gaps) and ClaimAssessmentRecords (the LLM assessors' per-claim
    missing_evidence, now persisted instead of discarded) into an actionable
    task list. The 3 assessors usually flag near-identical gaps for the same
    claim in slightly different words - deduped with lexical_similarity
    (already used elsewhere in this codebase for the same kind of fuzzy
    text matching) rather than 3x near-duplicate tasks per claim."""
    project_id = state["context"]["project_id"]
    run_id = state["context"]["run_id"]

    findings = repo.get_findings(project_id)
    claims_by_id = {c.claim_id: c for c in repo.get_claims(project_id)}
    assessments = repo.get_claim_assessments(project_id)

    tasks: list[CompletionTask] = []
    for finding in findings:
        if not finding.rule_id.startswith("missing_checklist_evidence:"):
            continue
        tasks.append(CompletionTask(
            task_id=new_id("TASK"), project_id=project_id, plan_id="", title=finding.message,
            reason=finding.message, category="MISSING_DATA", priority="REQUIRED", source=finding.finding_id,
        ))

    by_claim: dict[str, list[ClaimAssessmentRecord]] = {}
    for record in assessments:
        by_claim.setdefault(record.claim_id, []).append(record)

    for claim_id, records in by_claim.items():
        claim = claims_by_id.get(claim_id)
        priority = "REQUIRED" if claim and claim.importance == "CENTRAL" else "RECOMMENDED"
        distinct_items: list[str] = []
        for record in records:
            for item in record.missing_evidence:
                if not any(lexical_similarity(item, seen) >= 0.5 for seen in distinct_items):
                    distinct_items.append(item)
        for item in distinct_items:
            tasks.append(CompletionTask(
                task_id=new_id("TASK"), project_id=project_id, plan_id="", title=item,
                reason=f"Flagged for {claim_id}: {item}", category="VALIDATION", priority=priority,
                affected_claim_ids=[claim_id], source="claim_assessment",
            ))

    plan_id = new_id("COMPLETIONPLAN")
    tasks = [t.model_copy(update={"plan_id": plan_id}) for t in tasks]
    repo.add_completion_tasks(tasks)
    repo.add_completion_plan(CompletionPlan(
        plan_id=plan_id, project_id=project_id, run_id=run_id, task_ids=[t.task_id for t in tasks],
    ))

    return {
        "completion_plan_id": plan_id,
        "context": {
            **state["context"], "status": "SUCCEEDED",
            "current_stage": "COMPLETION_PLAN_READY", "updated_at": now_iso(),
        },
    }


_CLAIM_TYPE_SECTION_KEYWORD = {
    "RESULT": "results", "HYPOTHESIS": "introduction",
    "LITERATURE_CLAIM": "introduction", "INTERPRETATION": "discussion",
    "SUGGESTION": "discussion", "FACT": "introduction",
}
_DEFAULT_SECTIONS = ["Introduction", "Methods", "Results", "Discussion", "Conclusion"]
_FRONT_BACK_MATTER = {
    "title", "abstract", "references", "acknowledgements",
    "conflicts_of_interest", "supporting_information",
}


def _pick_section(section_names: list[str], keyword: str, fallback: str) -> str:
    for name in section_names:
        if keyword and keyword in name.lower():
            return name
    return fallback


def create_manuscript_plan_node(state: AssessmentState, repo: Repository) -> dict[str, Any]:
    """Section list comes from the target journal's real required_sections
    (falls back to a plain IMRaD list when no journal was given). Claims are
    allocated to a section by claim_type via keyword match against the
    actual section names, not exact equality - a profile can combine
    "results_and_discussion" into one section, and this still finds it."""
    project_id = state["context"]["project_id"]
    run_id = state["context"]["run_id"]
    journal_id = state.get("target_journal_id")

    profile = load_target_journal_profile(journal_id)
    section_names = [
        s.replace("_", " ").title()
        for s in (profile or {}).get("manuscript_structure", {}).get("required_sections", [])
        if s not in _FRONT_BACK_MATTER
    ] or list(_DEFAULT_SECTIONS)

    claims = repo.get_claims(project_id)
    by_section: dict[str, list[str]] = {name: [] for name in section_names}
    fallback_section = section_names[-1]
    for claim in claims:
        keyword = _CLAIM_TYPE_SECTION_KEYWORD.get(claim.claim_type, "")
        target = _pick_section(section_names, keyword, fallback_section)
        by_section[target].append(claim.claim_id)

    plan_id = new_id("MSPLAN")
    sections = [
        PlannedSection(
            section_id=new_id("SECTION"), project_id=project_id, plan_id=plan_id,
            name=name, order=i, claim_ids=by_section[name],
        )
        for i, name in enumerate(section_names)
    ]
    repo.add_planned_sections(sections)
    repo.add_manuscript_plan(ManuscriptPlan(
        plan_id=plan_id, project_id=project_id, run_id=run_id, journal_id=journal_id,
        section_ids=[s.section_id for s in sections],
    ))

    return {
        "manuscript_plan_id": plan_id,
        "context": {
            **state["context"], "status": "SUCCEEDED",
            "current_stage": "MANUSCRIPT_PLAN_READY", "updated_at": now_iso(),
        },
    }


def reassessment_terminal(state: AssessmentState) -> dict[str, Any]:
    return {
        "context": {
            **state["context"], "status": "SUCCEEDED",
            "current_stage": "REASSESSMENT_REQUESTED", "updated_at": now_iso(),
        }
    }


def blocked_terminal(state: AssessmentState) -> dict[str, Any]:
    return {
        "context": {**state["context"], "status": "BLOCKED", "current_stage": "BLOCKED", "updated_at": now_iso()}
    }


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def build_assessment_graph(repo: Repository, checkpoint_db_path: str | None = None):
    """checkpoint_db_path=None compiles without a checkpointer attached -
    used by studio_entry.py, since LangGraph's own dev/API server manages
    persistence itself and rejects a graph compiled with a custom one already
    attached. Real runs (cli.py, tests) pass a real path so our SqliteSaver
    is used instead."""
    builder = StateGraph(AssessmentState)

    builder.add_node("initialize_run", initialize_run)
    builder.add_node("validate_boundary", validate_boundary)
    builder.add_node("create_manifest", lambda s: create_manifest(s, repo))
    builder.add_node("discover_project", lambda s: discover_project(s, repo))
    builder.add_node("classify_domain", lambda s: classify_domain(s, repo))
    builder.add_node("run_audits", lambda s: run_audits(s, repo))
    builder.add_node("synthesize_readiness", lambda s: synthesize_readiness(s, repo))
    builder.add_node("human_assessment_review", lambda s: human_assessment_review(s, repo))
    builder.add_node("create_completion_plan", lambda s: create_completion_plan_node(s, repo))
    builder.add_node("create_manuscript_plan", lambda s: create_manuscript_plan_node(s, repo))
    builder.add_node("reassessment_requested", reassessment_terminal)
    builder.add_node("blocked", blocked_terminal)

    builder.add_edge(START, "initialize_run")
    builder.add_edge("initialize_run", "validate_boundary")
    builder.add_conditional_edges(
        "validate_boundary",
        lambda state: "continue" if not state.get("error") else "blocked",
        {"continue": "create_manifest", "blocked": "blocked"},
    )
    builder.add_edge("create_manifest", "discover_project")
    builder.add_edge("discover_project", "classify_domain")
    builder.add_edge("classify_domain", "run_audits")
    builder.add_edge("run_audits", "synthesize_readiness")
    builder.add_edge("synthesize_readiness", "human_assessment_review")
    builder.add_conditional_edges(
        "human_assessment_review",
        route_after_assessment,
        {
            "manuscript": "create_manuscript_plan",
            "completion": "create_completion_plan",
            "reassess": "reassessment_requested",
            "blocked": "blocked",
        },
    )
    builder.add_edge("create_manuscript_plan", END)
    builder.add_edge("create_completion_plan", END)
    builder.add_edge("reassessment_requested", END)
    builder.add_edge("blocked", END)

    if checkpoint_db_path is None:
        return builder.compile()
    return builder.compile(checkpointer=build_checkpointer(checkpoint_db_path))


def initial_state(
    *, source_path: str, project_id: str, run_id: str, thread_id: str, target_journal_id: str | None = None,
) -> AssessmentState:
    return {
        "context": {
            "project_id": project_id,
            "run_id": run_id,
            "thread_id": thread_id,
            "workflow_name": "project_assessment",
            "status": "CREATED",
            "current_stage": "CREATED",
            "created_at": now_iso(),
            "updated_at": now_iso(),
        },
        "source_path": source_path,
        "target_journal_id": target_journal_id,
        "audit_ids": [],
        "finding_ids": [],
        "candidate_claim_ids": [],
        "blocking_finding_ids": [],
    }
