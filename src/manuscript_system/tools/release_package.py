from __future__ import annotations

import hashlib

from ..domain.models import ReleaseCandidate
from ..persistence.repositories import Repository, new_id

"""Assembles everything a project has actually produced into one exportable
Markdown snapshot - a plain function, not a LangGraph graph, since this is
pure deterministic assembly of already-persisted data with no interrupts, no
decision points, and no LLM calls.

Deliberately does not attempt a cover letter, graphical abstract, supplement,
rendered LaTeX/DOCX/PDF, or "environment information" - none of that is
backed by anything real in this codebase yet (no cover-letter prompt, no
image generation, no document renderer, no environment-capture), and faking
it would produce empty placeholders rather than a real feature.

No hard "final gate" check before assembly - there are multiple possible
"final" states depending on which path a project took (completion plan vs.
manuscript plan vs. full draft+review cycle), no single state to gate on
without guessing wrong for some path. Instead the rendered document
prominently surfaces current readiness/approval status so a human glancing
at it immediately sees whether it's actually ready.
"""


def _peer_review_detail(repo: Repository, project_id: str, checkpoint_db_path: str | None):
    """The PeerReviewRound summary alone doesn't carry the 5 reviewer
    reports, revision plan, or proposals - those live only in the review
    graph's own LangGraph checkpoint. Read them back via get_state rather
    than denormalizing everything into PeerReviewRound."""
    rounds = repo.get_peer_review_rounds(project_id)
    if not rounds:
        return None, None
    latest = rounds[-1]
    from ..graphs.manuscript_review import build_manuscript_review_graph
    graph = build_manuscript_review_graph(checkpoint_db_path)
    snapshot = graph.get_state({"configurable": {"thread_id": latest.thread_id}})
    return latest, snapshot.values if snapshot else None


def assemble_release_package(
    repo: Repository, project_id: str, checkpoint_db_path: str | None = None,
) -> tuple[ReleaseCandidate, str]:
    project = repo.get_project(project_id)
    plan = repo.get_manuscript_plan(project_id)
    sections = repo.get_planned_sections(project_id) if plan else []
    blocks = repo.get_manuscript_blocks(project_id)
    blocks_by_section = {b.section_id: b for b in blocks}
    readiness = repo.get_readiness_report(project_id)
    completion_plan = repo.get_completion_plan(project_id)
    completion_tasks = repo.get_completion_tasks(project_id) if completion_plan else []
    peer_review_round, review_state = _peer_review_detail(repo, project_id, checkpoint_db_path)
    citations = repo.get_citations(project_id)
    findings = repo.get_findings(project_id)
    claims = repo.get_claims(project_id)
    evidence_by_id = {e.evidence_id: e for e in repo.get_evidence_items(project_id)}
    human_decisions = repo.get_human_decisions(project_id)

    lines: list[str] = []
    if project:
        lines.append(f"Source project: `{project.source_path}`")
        lines.append("")

    lines.append("## Readiness")
    if readiness:
        lines.append(f"**Status: {readiness.readiness_status}**")
        lines.append("")
        lines.append(readiness.explanation or "*(no explanation recorded)*")
    else:
        lines.append("*(no readiness report found for this project)*")
    lines.append("")

    lines.append("## Manuscript")
    if not plan:
        lines.append("*(no approved manuscript plan for this project)*")
        lines.append("")
    else:
        for section in sorted(sections, key=lambda s: s.order):
            lines.append(f"### {section.name}")
            block = blocks_by_section.get(section.section_id)
            if block is None:
                lines.append("*(not drafted — no claims allocated to this section)*")
            else:
                lines.append(block.text)
                if block.semantic_warnings:
                    lines.append("")
                    lines.append(f"> Warnings: {'; '.join(block.semantic_warnings)}")
            lines.append("")

    lines.append("## Claim / Evidence Map")
    if not claims:
        lines.append("*(no claims recorded)*")
    else:
        for claim in claims:
            lines.append(f"- **{claim.claim_id}** ({claim.claim_type}, {claim.importance}): {claim.text}")
            for evidence_id in claim.evidence_ids:
                item = evidence_by_id.get(evidence_id)
                if item:
                    lines.append(f"  - evidence `{item.evidence_id}` ({item.evidence_type}): {item.excerpt_or_value}")
    lines.append("")

    lines.append("## Completion Plan")
    if not completion_plan:
        lines.append("*(no completion plan for this project)*")
    elif not completion_tasks:
        lines.append("*(completion plan exists but has no tasks)*")
    else:
        for task in completion_tasks:
            lines.append(f"- [{task.priority}] ({task.category}) {task.title}")
    lines.append("")

    lines.append("## Peer Review")
    if not peer_review_round:
        lines.append("*(peer review has not been run for this project)*")
    else:
        lines.append(f"Round status: **{peer_review_round.status}**")
        lines.append("")
        reports = (review_state or {}).get("review_reports", [])
        if reports:
            lines.append("### Reviewer reports")
            for report in reports:
                lines.append(
                    f"- **{report.get('persona')}** — {report.get('overall_recommendation')}: "
                    f"{report.get('summary')}"
                )
                for comment in report.get("comments", []):
                    lines.append(f"  - ({comment.get('severity')}/{comment.get('category')}) {comment.get('text')}")
        revision_plan = (review_state or {}).get("revision_plan")
        if revision_plan:
            lines.append("")
            lines.append("### Revision history")
            for task in revision_plan.get("revision_tasks", []):
                lines.append(f"- {task}")
        response = peer_review_round.response_to_reviewers
        if response:
            lines.append("")
            lines.append("### Response to reviewers")
            for item in response.get("responses", []):
                lines.append(f"- [{item.get('comment_id')}] {item.get('status')}: {item.get('response', '')}")
    lines.append("")

    lines.append("## Bibliography")
    if not citations:
        lines.append(
            "*(no citations recorded for this project — citation verification runs under a separate "
            "`evaluate-manuscript` project, not automatically linked to this one)*"
        )
    else:
        for citation in citations:
            lines.append(f"- `{citation.cite_key}` ({citation.verification_status}): {citation.bib_title}")
    lines.append("")

    lines.append("## Findings")
    if not findings:
        lines.append("*(no findings recorded)*")
    else:
        for finding in findings:
            lines.append(f"- [{finding.severity}] ({finding.rule_id}) {finding.message}")
    lines.append("")

    lines.append("## Approval Audit")
    if not human_decisions:
        lines.append("*(no human decisions recorded)*")
    else:
        for decision in human_decisions:
            lines.append(f"- {decision.created_at} — **{decision.kind}**: {decision.decision}")
    lines.append("")

    body = "\n".join(lines)
    package_hash = hashlib.sha256(body.encode("utf-8")).hexdigest()

    candidate = ReleaseCandidate(
        release_id=new_id("RELEASE"), project_id=project_id,
        manuscript_plan_id=plan.plan_id if plan else None,
        readiness_report_id=readiness.report_id if readiness else None,
        completion_plan_id=completion_plan.plan_id if completion_plan else None,
        peer_review_round_id=peer_review_round.round_id if peer_review_round else None,
        section_count=len(sections), drafted_section_count=len(blocks_by_section),
        citation_count=len(citations), human_decision_count=len(human_decisions),
        package_hash=package_hash,
    )

    header = (
        f"# Manuscript Release Package — {project_id}\n\n"
        f"Generated: {candidate.created_at}  \nPackage hash: `{package_hash}`\n\n"
    )
    return candidate, header + body
