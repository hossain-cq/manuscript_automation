from __future__ import annotations

"""Researcher CLI - Phase 0 shape.

Ported/trimmed from the repo-root manuscriptctl.py prototype. That version was
an HTTP client for a control-plane API that doesn't exist yet; this version
calls the assessment graph in-process instead, so the CLI is usable today. The
`start`/`status`/`approve`/`resume`/`export` command names and shapes carry
over from manuscriptctl.py so a future HTTP-backed version is a drop-in swap.
"""

import argparse
import json
import sys
from typing import Any

from langgraph.types import Command

from .graphs.assessment import ASSESSMENT_CHOICES, build_assessment_graph, initial_state
from .graphs.manuscript_drafting import (
    DRAFT_CHOICES,
    build_manuscript_drafting_graph,
    build_section_specs,
    initial_state as drafting_initial_state,
)
from .graphs.manuscript_evaluation import (
    MANUSCRIPT_CHOICES,
    build_manuscript_evaluation_graph,
    initial_state as manuscript_initial_state,
)
from .graphs.manuscript_review import (
    REVIEW_CHOICES,
    _to_ported_journal_profile,
    build_manuscript_review_graph,
    initial_state as review_initial_state,
)
from .domain.models import PeerReviewRound
from .persistence.database import connect
from .persistence.repositories import Repository, new_id
from .settings import get_settings
from .tools.journals import load_journal_profile


def _print(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, default=str))


def cmd_intake(args: argparse.Namespace) -> None:
    settings = get_settings()
    repo = Repository(connect(settings.database_path))

    project = repo.create_project(source_path=args.path, domain_profile_id=args.domain)
    run_id = new_id("RUN")
    thread_id = f"thread-{run_id}"
    repo.create_run(project_id=project.project_id, thread_id=thread_id, workflow_name="project_assessment")

    graph = build_assessment_graph(repo, settings.checkpoint_path)
    config = {"configurable": {"thread_id": thread_id}}
    state = initial_state(
        source_path=args.path, project_id=project.project_id, run_id=run_id, thread_id=thread_id,
        target_journal_id=args.journal,
    )

    result = graph.invoke(state, config=config)
    pending = result.get("__interrupt__")

    print(f"project_id: {project.project_id}")
    print(f"run_id:     {run_id}")
    print(f"thread_id:  {thread_id}")
    if result.get("error"):
        print(f"BLOCKED: {result['error']['message']}")
        return

    findings = repo.get_findings(project.project_id)
    if findings:
        print()
        print("Findings:")
        for finding in findings:
            print(f"  [{finding.severity}] ({finding.rule_id}) {finding.message}")

    print()
    print(f"Readiness: {result.get('readiness_status', '(none - no evidence extracted)')}")
    explanation = result.get("readiness_explanation")
    if explanation:
        print(explanation)

    assessor_reports = repo.get_assessor_reports(project.project_id)
    claim_assessments = repo.get_claim_assessments(project.project_id)
    if assessor_reports:
        print()
        print("Assessors:")
        for report in assessor_reports:
            if report.abstain:
                print(f"  {report.assessor_id}: ABSTAINED (insufficient evidence to score)")
                continue
            print(
                f"  {report.assessor_id}  confidence={report.confidence:.2f}  "
                f"contribution={report.scientific_contribution:.2f}  "
                f"evidence_sufficiency={report.evidence_sufficiency:.2f}  "
                f"validation_strength={report.validation_strength:.2f}  "
                f"reproducibility={report.reproducibility:.2f}"
            )
            for risk in report.major_risks:
                print(f"      risk: {risk}")

    if claim_assessments:
        print()
        print("Per-claim assessment:")
        by_claim: dict[str, list] = {}
        for record in claim_assessments:
            by_claim.setdefault(record.claim_id, []).append(record)
        for claim_id, records in by_claim.items():
            print(f"  {claim_id}:")
            for record in records:
                print(f"    [{record.assessor_id}] {record.label} (score={record.score:.2f}): {record.reasoning}")
                if record.missing_evidence:
                    print(f"      missing evidence: {'; '.join(record.missing_evidence)}")
                if record.limitations:
                    print(f"      limitations: {'; '.join(record.limitations)}")

    if pending:
        print()
        print("Waiting for human review. Choices:", ", ".join(ASSESSMENT_CHOICES))
        print(f"Resume with: manuscript-system approve --thread-id {thread_id} --decision <choice>")


def cmd_approve(args: argparse.Namespace) -> None:
    settings = get_settings()
    repo = Repository(connect(settings.database_path))
    graph = build_assessment_graph(repo, settings.checkpoint_path)
    config = {"configurable": {"thread_id": args.thread_id}}

    result = graph.invoke(Command(resume={"decision": args.decision}), config=config)
    context = result.get("context", {})
    project_id = context.get("project_id")

    print(f"thread_id: {args.thread_id}")
    print(f"status:    {context.get('status')}  ({context.get('current_stage')})")

    if result.get("completion_plan_id") and project_id:
        tasks = repo.get_completion_tasks(project_id)
        print()
        print(f"Completion plan {result['completion_plan_id']} - {len(tasks)} task(s):")
        for task in tasks:
            claim_note = f" [{', '.join(task.affected_claim_ids)}]" if task.affected_claim_ids else ""
            print(f"  [{task.priority}] ({task.category}) {task.title}{claim_note}")

    if result.get("manuscript_plan_id") and project_id:
        sections = repo.get_planned_sections(project_id)
        print()
        print(f"Manuscript plan {result['manuscript_plan_id']} - {len(sections)} section(s):")
        for section in sections:
            claims_note = f" - {len(section.claim_ids)} claim(s): {', '.join(section.claim_ids)}" if section.claim_ids else " - no claims allocated"
            print(f"  {section.order + 1}. {section.name}{claims_note}")

    if not result.get("completion_plan_id") and not result.get("manuscript_plan_id"):
        _print({"context": context})


def cmd_evaluate_manuscript(args: argparse.Namespace) -> None:
    settings = get_settings()
    repo = Repository(connect(settings.database_path))

    project = repo.create_project(source_path=args.path, domain_profile_id=args.journal)
    run_id = new_id("RUN")
    thread_id = f"thread-{run_id}"
    repo.create_run(project_id=project.project_id, thread_id=thread_id, workflow_name="manuscript_evaluation")

    graph = build_manuscript_evaluation_graph(repo, settings.checkpoint_path)
    config = {"configurable": {"thread_id": thread_id}}
    state = manuscript_initial_state(
        manuscript_path=args.path, project_id=project.project_id, run_id=run_id,
        thread_id=thread_id, journal_id=args.journal, linked_data_path=args.data_path,
    )

    result = graph.invoke(state, config=config)
    pending = result.get("__interrupt__")

    print(f"project_id: {project.project_id}")
    print(f"run_id:     {run_id}")
    print(f"thread_id:  {thread_id}")
    if result.get("error"):
        print(f"BLOCKED: {result['error']['message']}")
        return
    print()
    print(result.get("summary", "(no summary produced)"))

    findings = repo.get_findings(project.project_id)
    if findings:
        print()
        print("Findings:")
        for finding in findings:
            print(f"  [{finding.severity}] ({finding.rule_id}) {finding.message}")

    if pending:
        print()
        print("Waiting for human review. Choices:", ", ".join(MANUSCRIPT_CHOICES))
        print(f"Resume with: manuscript-system approve-manuscript --thread-id {thread_id} --decision <choice>")


def cmd_approve_manuscript(args: argparse.Namespace) -> None:
    settings = get_settings()
    repo = Repository(connect(settings.database_path))
    graph = build_manuscript_evaluation_graph(repo, settings.checkpoint_path)
    config = {"configurable": {"thread_id": args.thread_id}}

    result = graph.invoke(Command(resume={"decision": args.decision}), config=config)
    _print({"thread_id": args.thread_id, "context": result.get("context")})


def cmd_draft(args: argparse.Namespace) -> None:
    settings = get_settings()
    repo = Repository(connect(settings.database_path))

    plan = repo.get_manuscript_plan(args.project_id)
    if plan is None:
        print(
            f"No manuscript plan found for project '{args.project_id}'. "
            "Run intake, then approve --decision APPROVE_MANUSCRIPT_PLANNING first."
        )
        return

    sections = repo.get_planned_sections(args.project_id)
    claims = repo.get_claims(args.project_id)
    evidence = repo.get_evidence_items(args.project_id)
    section_specs = build_section_specs(sections, claims)
    skipped = [s.name for s in sections if not s.claim_ids]

    run_id = new_id("RUN")
    thread_id = f"thread-{run_id}"
    repo.create_run(project_id=args.project_id, thread_id=thread_id, workflow_name="manuscript_drafting")

    print(f"project_id: {args.project_id}")
    print(f"plan_id:    {plan.plan_id}")
    print(f"thread_id:  {thread_id}")

    if skipped:
        print()
        print(f"Skipped (no claims allocated): {', '.join(skipped)}")

    if not section_specs:
        print()
        print("No sections with allocated claims to draft.")
        return

    graph = build_manuscript_drafting_graph(repo, args.project_id, plan.plan_id, settings.checkpoint_path)
    config = {"configurable": {"thread_id": thread_id}}
    state = drafting_initial_state(
        project_id=args.project_id, section_specs=section_specs, claims=claims, evidence=evidence,
    )
    result = graph.invoke(state, config=config)

    if result.get("draft_status") == "BLOCKED":
        print()
        print(f"BLOCKED: {result.get('error', '(no reason given)')}")
        return

    blocks = repo.get_manuscript_blocks(args.project_id)
    print()
    print(f"Drafted {len(blocks)} section(s):")
    for block in blocks:
        print(f"  [{block.section_id}] claims={block.claim_ids}")
        if block.semantic_warnings:
            print(f"    warnings: {'; '.join(block.semantic_warnings)}")
        preview = block.text[:300] + ("..." if len(block.text) > 300 else "")
        print(f"    {preview}")

    if result.get("__interrupt__"):
        print()
        print("Waiting for human review. Choices:", ", ".join(DRAFT_CHOICES))
        print(f"Resume with: manuscript-system approve-draft --thread-id {thread_id} --decision <choice>")


def cmd_approve_draft(args: argparse.Namespace) -> None:
    settings = get_settings()
    repo = Repository(connect(settings.database_path))
    graph = build_manuscript_drafting_graph(repo, checkpoint_db_path=settings.checkpoint_path)
    config = {"configurable": {"thread_id": args.thread_id}}

    result = graph.invoke(Command(resume={"decision": args.decision}), config=config)
    print(f"thread_id: {args.thread_id}")
    print(f"status:    {result.get('draft_status')}")


def _print_review_result(result: dict[str, Any]) -> None:
    reports = result.get("review_reports") or []
    if reports:
        print()
        print("Reviewer reports:")
        for report in reports:
            print(f"  [{report.get('persona')}] {report.get('overall_recommendation')} - {report.get('summary')}")
            for comment in report.get("comments", []):
                print(f"    ({comment.get('severity')}/{comment.get('category')}) {comment.get('text')}")

    revision_plan = result.get("revision_plan")
    if revision_plan:
        print()
        print(f"Revision plan {revision_plan.get('plan_id')}:")
        print(f"  accepted: {revision_plan.get('accepted_comment_ids')}")
        print(f"  rejected: {revision_plan.get('rejected_comment_ids')}")
        for task in revision_plan.get("revision_tasks", []):
            print(f"  - {task}")

    proposals = result.get("revision_proposals") or []
    verifications = {v["comment_id"]: v for v in (result.get("revision_verifications") or [])}
    if proposals:
        print()
        print("Revision proposals:")
        for proposal in proposals:
            verification = verifications.get(proposal.get("comment_id"), {})
            print(f"  [{proposal.get('comment_id')}] verification={verification.get('status', 'N/A')}")
            if proposal.get("abstain"):
                print(f"    abstained: {proposal.get('change_summary')}")
            else:
                print(f"    {proposal.get('revised_text', '')[:300]}")

    response = result.get("response_to_reviewers")
    if response:
        print()
        print(f"Response to reviewers ({response.get('round_id')}):")
        for item in response.get("responses", []):
            print(f"  [{item.get('comment_id')}] {item.get('status')}: {item.get('response', '')[:200]}")


def cmd_review(args: argparse.Namespace) -> None:
    settings = get_settings()
    repo = Repository(connect(settings.database_path))

    plan = repo.get_manuscript_plan(args.project_id)
    if plan is None:
        print(f"No manuscript plan found for project '{args.project_id}'.")
        return

    blocks = repo.get_manuscript_blocks(args.project_id)
    if not blocks:
        print(f"No drafted manuscript blocks found for project '{args.project_id}'. Run 'draft' first.")
        return

    claims = repo.get_claims(args.project_id)
    evidence = repo.get_evidence_items(args.project_id)
    profile = load_journal_profile(plan.journal_id)
    journal_profile = _to_ported_journal_profile(plan.journal_id, profile)

    run_id = new_id("RUN")
    thread_id = f"thread-{run_id}"
    repo.create_run(project_id=args.project_id, thread_id=thread_id, workflow_name="manuscript_review")

    print(f"project_id: {args.project_id}")
    print(f"plan_id:    {plan.plan_id}")
    print(f"thread_id:  {thread_id}")

    graph = build_manuscript_review_graph(settings.checkpoint_path)
    config = {"configurable": {"thread_id": thread_id}}
    state = review_initial_state(
        project_id=args.project_id, blocks=blocks, claims=claims, evidence=evidence,
        journal_profile=journal_profile,
    )
    result = graph.invoke(state, config=config)
    _print_review_result(result)

    if result.get("__interrupt__"):
        print()
        print("Waiting for human review. Choices:", ", ".join(REVIEW_CHOICES))
        print(f"Resume with: manuscript-system approve-review --thread-id {thread_id} --decision <choice>")


def cmd_approve_review(args: argparse.Namespace) -> None:
    settings = get_settings()
    repo = Repository(connect(settings.database_path))
    graph = build_manuscript_review_graph(settings.checkpoint_path)
    config = {"configurable": {"thread_id": args.thread_id}}

    result = graph.invoke(Command(resume={"decision": args.decision}), config=config)
    print(f"thread_id: {args.thread_id}")
    print(f"status:    {result.get('terminal_status')}")
    _print_review_result(result)

    if result.get("terminal_status") in {"SUCCEEDED", "BLOCKED"}:
        run = repo.get_run_by_thread_id(args.thread_id)
        if run is not None:
            plan = repo.get_manuscript_plan(run.project_id)
            repo.add_peer_review_round(PeerReviewRound(
                round_id=new_id("REVIEWROUND"), project_id=run.project_id,
                plan_id=plan.plan_id if plan else "", thread_id=args.thread_id,
                status=result.get("terminal_status", ""), response_to_reviewers=result.get("response_to_reviewers") or {},
            ))
            print()
            print("Peer review round persisted.")

    if result.get("__interrupt__"):
        print()
        print("Waiting for human review. Choices:", ", ".join(REVIEW_CHOICES))
        print(f"Resume with: manuscript-system approve-review --thread-id {args.thread_id} --decision <choice>")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="manuscript-system")
    sub = parser.add_subparsers(dest="command", required=True)

    intake = sub.add_parser(
        "intake", help="register a project and run the assessment graph up to the first human gate"
    )
    intake.add_argument("--path", required=True, help="absolute path to the project folder")
    intake.add_argument("--domain", default="quantum_chemistry")
    intake.add_argument(
        "--journal", default=None,
        help="target journal profile id from configs/journals.yaml - used to shape the manuscript "
             "plan's section outline if the assessment is later approved for manuscript planning",
    )
    intake.add_argument(
        "--read-only", action="store_true", default=True,
        help="Phase 0 never executes project code; kept for CLI-shape compatibility with the eventual API",
    )
    intake.set_defaults(func=cmd_intake)

    approve = sub.add_parser("approve", help="resume a run waiting on a human decision")
    approve.add_argument("--thread-id", required=True)
    approve.add_argument("--decision", required=True, choices=list(ASSESSMENT_CHOICES))
    approve.set_defaults(func=cmd_approve)

    evaluate = sub.add_parser(
        "evaluate-manuscript",
        help="evaluate an already-written manuscript (citation integrity + journal compliance + optional data cross-check)",
    )
    evaluate.add_argument("--path", required=True, help="absolute path to the manuscript folder")
    evaluate.add_argument("--journal", default=None, help="journal profile id from configs/journals.yaml")
    evaluate.add_argument(
        "--data-path", default=None,
        help="absolute path to the linked raw-data project - if given, numeric claims in the manuscript "
             "are cross-checked against real values in that project's .csv/.dat/.ipynb files",
    )
    evaluate.set_defaults(func=cmd_evaluate_manuscript)

    approve_manuscript = sub.add_parser("approve-manuscript", help="resume a manuscript evaluation run")
    approve_manuscript.add_argument("--thread-id", required=True)
    approve_manuscript.add_argument("--decision", required=True, choices=list(MANUSCRIPT_CHOICES))
    approve_manuscript.set_defaults(func=cmd_approve_manuscript)

    draft = sub.add_parser(
        "draft",
        help="draft manuscript sections (real LLM calls) from an approved ManuscriptPlan's claim allocations",
    )
    draft.add_argument("--project-id", required=True, help="project id from an approved MANUSCRIPT_PLANNING run")
    draft.set_defaults(func=cmd_draft)

    approve_draft = sub.add_parser("approve-draft", help="resume a drafting run waiting on human review")
    approve_draft.add_argument("--thread-id", required=True)
    approve_draft.add_argument("--decision", required=True, choices=list(DRAFT_CHOICES))
    approve_draft.set_defaults(func=cmd_approve_draft)

    review = sub.add_parser(
        "review",
        help="simulate peer review (real LLM calls, 5 personas) against a project's drafted manuscript blocks",
    )
    review.add_argument("--project-id", required=True, help="project id with drafted manuscript blocks")
    review.set_defaults(func=cmd_review)

    approve_review = sub.add_parser("approve-review", help="resume a review run waiting on one of its 3 human gates")
    approve_review.add_argument("--thread-id", required=True)
    approve_review.add_argument(
        "--decision", required=True, choices=list(REVIEW_CHOICES),
        help="the review graph has 3 human gates with different choice sets - pass whichever this thread is paused at",
    )
    approve_review.set_defaults(func=cmd_approve_review)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
