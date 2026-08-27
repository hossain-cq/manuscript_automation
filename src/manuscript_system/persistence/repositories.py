from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone

from ..domain.models import (
    AssessorReport,
    Citation,
    Claim,
    ClaimAssessmentRecord,
    CompletionPlan,
    CompletionTask,
    Evidence,
    Finding,
    HumanDecision,
    ManuscriptBlock,
    ManuscriptPlan,
    PeerReviewRound,
    NumericCrossCheck,
    PlannedSection,
    Project,
    ProjectKnowledgeMap,
    ReadinessReport,
    ReleaseCandidate,
    Run,
    SourceAsset,
)


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Repository:
    """SQLite-backed repository shared across the graphs.

    Replaces the per-module InMemoryRepository / ManuscriptRepository classes
    each prototype graph file previously defined for itself, and the fake
    persist_* helper functions supervisor_graph.py used as placeholders.
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    def create_project(self, *, source_path: str, domain_profile_id: str | None = None) -> Project:
        project = Project(
            project_id=new_id("PROJECT"), source_path=source_path, domain_profile_id=domain_profile_id
        )
        self.conn.execute(
            "INSERT INTO projects (project_id, source_path, domain_profile_id, created_at) "
            "VALUES (?, ?, ?, ?)",
            (project.project_id, project.source_path, project.domain_profile_id, project.created_at),
        )
        self.conn.commit()
        return project

    def get_project(self, project_id: str) -> Project | None:
        row = self.conn.execute("SELECT * FROM projects WHERE project_id = ?", (project_id,)).fetchone()
        if row is None:
            return None
        return Project(
            project_id=row["project_id"], source_path=row["source_path"],
            domain_profile_id=row["domain_profile_id"], created_at=row["created_at"],
        )

    def add_source_assets(self, assets: list[SourceAsset]) -> None:
        if not assets:
            return
        self.conn.executemany(
            "INSERT OR REPLACE INTO source_assets "
            "(artifact_id, project_id, relative_path, checksum_sha256, size_bytes, media_type, modified_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (a.artifact_id, a.project_id, a.relative_path, a.checksum_sha256,
                 a.size_bytes, a.media_type, a.modified_at)
                for a in assets
            ],
        )
        self.conn.commit()

    def get_source_assets(self, project_id: str) -> list[SourceAsset]:
        rows = self.conn.execute(
            "SELECT artifact_id, project_id, relative_path, checksum_sha256, size_bytes, "
            "media_type, modified_at FROM source_assets WHERE project_id = ?",
            (project_id,),
        ).fetchall()
        return [
            SourceAsset(
                artifact_id=row["artifact_id"], project_id=row["project_id"],
                relative_path=row["relative_path"], checksum_sha256=row["checksum_sha256"],
                size_bytes=row["size_bytes"], media_type=row["media_type"],
                modified_at=row["modified_at"],
            )
            for row in rows
        ]

    def add_claims(self, claims: list[Claim]) -> None:
        if not claims:
            return
        self.conn.executemany(
            "INSERT OR REPLACE INTO claims "
            "(claim_id, project_id, text, claim_type, importance, evidence_ids, status) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (c.claim_id, c.project_id, c.text, c.claim_type, c.importance,
                 json.dumps(c.evidence_ids), c.status)
                for c in claims
            ],
        )
        self.conn.commit()

    def get_claims(self, project_id: str) -> list[Claim]:
        rows = self.conn.execute(
            "SELECT * FROM claims WHERE project_id = ? ORDER BY claim_id", (project_id,)
        ).fetchall()
        return [
            Claim(
                claim_id=row["claim_id"], project_id=row["project_id"], text=row["text"],
                claim_type=row["claim_type"], importance=row["importance"],
                evidence_ids=json.loads(row["evidence_ids"]), status=row["status"],
            )
            for row in rows
        ]

    def add_evidence_items(self, items: list[Evidence]) -> None:
        if not items:
            return
        self.conn.executemany(
            "INSERT OR REPLACE INTO evidence_items "
            "(evidence_id, project_id, source_artifact_id, evidence_type, location, "
            "excerpt_or_value, extraction_confidence) VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (e.evidence_id, e.project_id, e.source_artifact_id, e.evidence_type,
                 e.location, e.excerpt_or_value, e.extraction_confidence)
                for e in items
            ],
        )
        self.conn.commit()

    def get_evidence_items(self, project_id: str) -> list[Evidence]:
        rows = self.conn.execute(
            "SELECT * FROM evidence_items WHERE project_id = ? ORDER BY evidence_id", (project_id,)
        ).fetchall()
        return [
            Evidence(
                evidence_id=row["evidence_id"], project_id=row["project_id"],
                source_artifact_id=row["source_artifact_id"], evidence_type=row["evidence_type"],
                location=row["location"], excerpt_or_value=row["excerpt_or_value"],
                extraction_confidence=row["extraction_confidence"],
            )
            for row in rows
        ]

    def add_citations(self, citations: list[Citation]) -> None:
        if not citations:
            return
        self.conn.executemany(
            "INSERT OR REPLACE INTO citations "
            "(citation_id, project_id, cite_key, bib_title, bib_authors, bib_year, bib_doi, "
            "verification_status, verified_title, verified_doi, match_confidence, notes) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (c.citation_id, c.project_id, c.cite_key, c.bib_title, c.bib_authors, c.bib_year,
                 c.bib_doi, c.verification_status, c.verified_title, c.verified_doi,
                 c.match_confidence, c.notes)
                for c in citations
            ],
        )
        self.conn.commit()

    def get_citations(self, project_id: str) -> list[Citation]:
        rows = self.conn.execute(
            "SELECT * FROM citations WHERE project_id = ? ORDER BY cite_key", (project_id,)
        ).fetchall()
        return [
            Citation(
                citation_id=row["citation_id"], project_id=row["project_id"], cite_key=row["cite_key"],
                bib_title=row["bib_title"], bib_authors=row["bib_authors"], bib_year=row["bib_year"],
                bib_doi=row["bib_doi"], verification_status=row["verification_status"],
                verified_title=row["verified_title"], verified_doi=row["verified_doi"],
                match_confidence=row["match_confidence"], notes=row["notes"],
            )
            for row in rows
        ]

    def add_assessor_reports(self, reports: list[AssessorReport]) -> None:
        if not reports:
            return
        self.conn.executemany(
            "INSERT OR REPLACE INTO assessor_reports "
            "(report_id, project_id, run_id, assessor_id, scientific_contribution, evidence_sufficiency, "
            "methodological_rigor, validation_strength, reproducibility, literature_positioning, "
            "potential_significance, major_risks, contribution_candidates, confidence, abstain, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (r.report_id, r.project_id, r.run_id, r.assessor_id, r.scientific_contribution,
                 r.evidence_sufficiency, r.methodological_rigor, r.validation_strength, r.reproducibility,
                 r.literature_positioning, r.potential_significance, json.dumps(r.major_risks),
                 json.dumps(r.contribution_candidates), r.confidence, int(r.abstain), r.created_at)
                for r in reports
            ],
        )
        self.conn.commit()

    def get_assessor_reports(self, project_id: str) -> list[AssessorReport]:
        rows = self.conn.execute(
            "SELECT * FROM assessor_reports WHERE project_id = ? ORDER BY assessor_id", (project_id,)
        ).fetchall()
        return [
            AssessorReport(
                report_id=row["report_id"], project_id=row["project_id"], run_id=row["run_id"],
                assessor_id=row["assessor_id"], scientific_contribution=row["scientific_contribution"],
                evidence_sufficiency=row["evidence_sufficiency"], methodological_rigor=row["methodological_rigor"],
                validation_strength=row["validation_strength"], reproducibility=row["reproducibility"],
                literature_positioning=row["literature_positioning"],
                potential_significance=row["potential_significance"],
                major_risks=json.loads(row["major_risks"]), contribution_candidates=json.loads(row["contribution_candidates"]),
                confidence=row["confidence"], abstain=bool(row["abstain"]), created_at=row["created_at"],
            )
            for row in rows
        ]

    def add_claim_assessments(self, records: list[ClaimAssessmentRecord]) -> None:
        if not records:
            return
        self.conn.executemany(
            "INSERT OR REPLACE INTO claim_assessments "
            "(record_id, project_id, run_id, assessor_id, claim_id, label, score, reasoning, "
            "limitations, missing_evidence, confidence) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (r.record_id, r.project_id, r.run_id, r.assessor_id, r.claim_id, r.label, r.score,
                 r.reasoning, json.dumps(r.limitations), json.dumps(r.missing_evidence), r.confidence)
                for r in records
            ],
        )
        self.conn.commit()

    def get_claim_assessments(self, project_id: str) -> list[ClaimAssessmentRecord]:
        rows = self.conn.execute(
            "SELECT * FROM claim_assessments WHERE project_id = ? ORDER BY claim_id, assessor_id", (project_id,)
        ).fetchall()
        return [
            ClaimAssessmentRecord(
                record_id=row["record_id"], project_id=row["project_id"], run_id=row["run_id"],
                assessor_id=row["assessor_id"], claim_id=row["claim_id"], label=row["label"],
                score=row["score"], reasoning=row["reasoning"], limitations=json.loads(row["limitations"]),
                missing_evidence=json.loads(row["missing_evidence"]), confidence=row["confidence"],
            )
            for row in rows
        ]

    def add_numeric_cross_checks(self, checks: list[NumericCrossCheck]) -> None:
        if not checks:
            return
        self.conn.executemany(
            "INSERT OR REPLACE INTO numeric_cross_checks "
            "(check_id, project_id, claimed_value, claimed_text, source_section, status, "
            "matched_value, matched_source_path, matched_context, precision_places) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (c.check_id, c.project_id, c.claimed_value, c.claimed_text, c.source_section, c.status,
                 c.matched_value, c.matched_source_path, c.matched_context, c.precision_places)
                for c in checks
            ],
        )
        self.conn.commit()

    def get_numeric_cross_checks(self, project_id: str) -> list[NumericCrossCheck]:
        rows = self.conn.execute(
            "SELECT * FROM numeric_cross_checks WHERE project_id = ? ORDER BY claimed_value", (project_id,)
        ).fetchall()
        return [
            NumericCrossCheck(
                check_id=row["check_id"], project_id=row["project_id"], claimed_value=row["claimed_value"],
                claimed_text=row["claimed_text"], source_section=row["source_section"], status=row["status"],
                matched_value=row["matched_value"], matched_source_path=row["matched_source_path"],
                matched_context=row["matched_context"], precision_places=row["precision_places"],
            )
            for row in rows
        ]

    def create_run(self, *, project_id: str, thread_id: str, workflow_name: str) -> Run:
        run = Run(
            run_id=new_id("RUN"), project_id=project_id, thread_id=thread_id,
            workflow_name=workflow_name, status="CREATED", current_stage="CREATED",
        )
        self.conn.execute(
            "INSERT INTO runs "
            "(run_id, project_id, thread_id, workflow_name, status, current_stage, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (run.run_id, run.project_id, run.thread_id, run.workflow_name,
             run.status, run.current_stage, run.created_at, run.updated_at),
        )
        self.conn.commit()
        return run

    def get_run_by_thread_id(self, thread_id: str) -> Run | None:
        row = self.conn.execute("SELECT * FROM runs WHERE thread_id = ?", (thread_id,)).fetchone()
        if row is None:
            return None
        return Run(
            run_id=row["run_id"], project_id=row["project_id"], thread_id=row["thread_id"],
            workflow_name=row["workflow_name"], status=row["status"], current_stage=row["current_stage"],
            created_at=row["created_at"], updated_at=row["updated_at"],
        )

    def update_run_status(self, run_id: str, *, status: str, current_stage: str) -> None:
        self.conn.execute(
            "UPDATE runs SET status = ?, current_stage = ?, updated_at = ? WHERE run_id = ?",
            (status, current_stage, utc_now(), run_id),
        )
        self.conn.commit()

    def add_finding(self, finding: Finding) -> str:
        self.conn.execute(
            "INSERT INTO findings "
            "(finding_id, project_id, severity, rule_id, message, blocking, affected_claim_ids) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (finding.finding_id, finding.project_id, finding.severity, finding.rule_id,
             finding.message, int(finding.blocking), json.dumps(finding.affected_claim_ids)),
        )
        self.conn.commit()
        return finding.finding_id

    def get_findings(self, project_id: str) -> list[Finding]:
        rows = self.conn.execute(
            "SELECT * FROM findings WHERE project_id = ? ORDER BY finding_id", (project_id,)
        ).fetchall()
        return [
            Finding(
                finding_id=row["finding_id"], project_id=row["project_id"], severity=row["severity"],
                rule_id=row["rule_id"], message=row["message"], blocking=bool(row["blocking"]),
                affected_claim_ids=json.loads(row["affected_claim_ids"]),
            )
            for row in rows
        ]

    def add_knowledge_map(self, knowledge_map: ProjectKnowledgeMap) -> str:
        self.conn.execute(
            "INSERT INTO project_knowledge_maps "
            "(map_id, project_id, run_id, asset_counts_by_role, total_assets, total_size_bytes, "
            "has_readme, has_environment_file, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (knowledge_map.map_id, knowledge_map.project_id, knowledge_map.run_id,
             json.dumps(knowledge_map.asset_counts_by_role), knowledge_map.total_assets,
             knowledge_map.total_size_bytes, int(knowledge_map.has_readme),
             int(knowledge_map.has_environment_file), knowledge_map.created_at),
        )
        self.conn.commit()
        return knowledge_map.map_id

    def get_knowledge_map(self, project_id: str) -> ProjectKnowledgeMap | None:
        row = self.conn.execute(
            "SELECT * FROM project_knowledge_maps WHERE project_id = ? ORDER BY created_at DESC LIMIT 1",
            (project_id,),
        ).fetchone()
        if row is None:
            return None
        return ProjectKnowledgeMap(
            map_id=row["map_id"], project_id=row["project_id"], run_id=row["run_id"],
            asset_counts_by_role=json.loads(row["asset_counts_by_role"]), total_assets=row["total_assets"],
            total_size_bytes=row["total_size_bytes"], has_readme=bool(row["has_readme"]),
            has_environment_file=bool(row["has_environment_file"]), created_at=row["created_at"],
        )

    def add_completion_tasks(self, tasks: list[CompletionTask]) -> None:
        if not tasks:
            return
        self.conn.executemany(
            "INSERT OR REPLACE INTO completion_tasks "
            "(task_id, project_id, plan_id, title, reason, category, priority, affected_claim_ids, source) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (t.task_id, t.project_id, t.plan_id, t.title, t.reason, t.category, t.priority,
                 json.dumps(t.affected_claim_ids), t.source)
                for t in tasks
            ],
        )
        self.conn.commit()

    def get_completion_tasks(self, project_id: str) -> list[CompletionTask]:
        rows = self.conn.execute(
            "SELECT * FROM completion_tasks WHERE project_id = ? ORDER BY priority, task_id", (project_id,)
        ).fetchall()
        return [
            CompletionTask(
                task_id=row["task_id"], project_id=row["project_id"], plan_id=row["plan_id"],
                title=row["title"], reason=row["reason"], category=row["category"], priority=row["priority"],
                affected_claim_ids=json.loads(row["affected_claim_ids"]), source=row["source"],
            )
            for row in rows
        ]

    def add_completion_plan(self, plan: CompletionPlan) -> str:
        self.conn.execute(
            "INSERT INTO completion_plans (plan_id, project_id, run_id, task_ids, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (plan.plan_id, plan.project_id, plan.run_id, json.dumps(plan.task_ids), plan.created_at),
        )
        self.conn.commit()
        return plan.plan_id

    def get_completion_plan(self, project_id: str) -> CompletionPlan | None:
        row = self.conn.execute(
            "SELECT * FROM completion_plans WHERE project_id = ? ORDER BY created_at DESC LIMIT 1", (project_id,)
        ).fetchone()
        if row is None:
            return None
        return CompletionPlan(
            plan_id=row["plan_id"], project_id=row["project_id"], run_id=row["run_id"],
            task_ids=json.loads(row["task_ids"]), created_at=row["created_at"],
        )

    def add_planned_sections(self, sections: list[PlannedSection]) -> None:
        if not sections:
            return
        self.conn.executemany(
            'INSERT OR REPLACE INTO planned_sections '
            '(section_id, project_id, plan_id, name, "order", claim_ids) VALUES (?, ?, ?, ?, ?, ?)',
            [
                (s.section_id, s.project_id, s.plan_id, s.name, s.order, json.dumps(s.claim_ids))
                for s in sections
            ],
        )
        self.conn.commit()

    def get_planned_sections(self, project_id: str) -> list[PlannedSection]:
        rows = self.conn.execute(
            'SELECT * FROM planned_sections WHERE project_id = ? ORDER BY "order"', (project_id,)
        ).fetchall()
        return [
            PlannedSection(
                section_id=row["section_id"], project_id=row["project_id"], plan_id=row["plan_id"],
                name=row["name"], order=row["order"], claim_ids=json.loads(row["claim_ids"]),
            )
            for row in rows
        ]

    def add_manuscript_plan(self, plan: ManuscriptPlan) -> str:
        self.conn.execute(
            "INSERT INTO manuscript_plans (plan_id, project_id, run_id, journal_id, section_ids, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (plan.plan_id, plan.project_id, plan.run_id, plan.journal_id,
             json.dumps(plan.section_ids), plan.created_at),
        )
        self.conn.commit()
        return plan.plan_id

    def get_manuscript_plan(self, project_id: str) -> ManuscriptPlan | None:
        row = self.conn.execute(
            "SELECT * FROM manuscript_plans WHERE project_id = ? ORDER BY created_at DESC LIMIT 1", (project_id,)
        ).fetchone()
        if row is None:
            return None
        return ManuscriptPlan(
            plan_id=row["plan_id"], project_id=row["project_id"], run_id=row["run_id"],
            journal_id=row["journal_id"], section_ids=json.loads(row["section_ids"]), created_at=row["created_at"],
        )

    def add_manuscript_block(self, block: ManuscriptBlock) -> str:
        self.conn.execute(
            "INSERT INTO manuscript_blocks "
            "(block_id, project_id, plan_id, section_id, text, claim_ids, evidence_ids, literature_ids, "
            "authoring_agent, prompt_hash, model_id, code_revision, status, semantic_warnings, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (block.block_id, block.project_id, block.plan_id, block.section_id, block.text,
             json.dumps(block.claim_ids), json.dumps(block.evidence_ids), json.dumps(block.literature_ids),
             block.authoring_agent, block.prompt_hash, block.model_id, block.code_revision, block.status,
             json.dumps(block.semantic_warnings), block.created_at),
        )
        self.conn.commit()
        return block.block_id

    def get_manuscript_blocks(self, project_id: str) -> list[ManuscriptBlock]:
        rows = self.conn.execute(
            "SELECT * FROM manuscript_blocks WHERE project_id = ? ORDER BY created_at", (project_id,)
        ).fetchall()
        return [
            ManuscriptBlock(
                block_id=row["block_id"], project_id=row["project_id"], plan_id=row["plan_id"],
                section_id=row["section_id"], text=row["text"], claim_ids=json.loads(row["claim_ids"]),
                evidence_ids=json.loads(row["evidence_ids"]), literature_ids=json.loads(row["literature_ids"]),
                authoring_agent=row["authoring_agent"], prompt_hash=row["prompt_hash"], model_id=row["model_id"],
                code_revision=row["code_revision"], status=row["status"],
                semantic_warnings=json.loads(row["semantic_warnings"]), created_at=row["created_at"],
            )
            for row in rows
        ]

    def add_peer_review_round(self, round_: PeerReviewRound) -> str:
        self.conn.execute(
            "INSERT INTO peer_review_rounds "
            "(round_id, project_id, plan_id, thread_id, status, response_to_reviewers, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (round_.round_id, round_.project_id, round_.plan_id, round_.thread_id, round_.status,
             json.dumps(round_.response_to_reviewers), round_.created_at),
        )
        self.conn.commit()
        return round_.round_id

    def get_peer_review_rounds(self, project_id: str) -> list[PeerReviewRound]:
        rows = self.conn.execute(
            "SELECT * FROM peer_review_rounds WHERE project_id = ? ORDER BY created_at", (project_id,)
        ).fetchall()
        return [
            PeerReviewRound(
                round_id=row["round_id"], project_id=row["project_id"], plan_id=row["plan_id"],
                thread_id=row["thread_id"], status=row["status"],
                response_to_reviewers=json.loads(row["response_to_reviewers"]), created_at=row["created_at"],
            )
            for row in rows
        ]

    def add_release_candidate(self, candidate: ReleaseCandidate) -> str:
        self.conn.execute(
            "INSERT INTO release_candidates "
            "(release_id, project_id, manuscript_plan_id, readiness_report_id, completion_plan_id, "
            "peer_review_round_id, section_count, drafted_section_count, citation_count, "
            "human_decision_count, package_hash, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (candidate.release_id, candidate.project_id, candidate.manuscript_plan_id,
             candidate.readiness_report_id, candidate.completion_plan_id, candidate.peer_review_round_id,
             candidate.section_count, candidate.drafted_section_count, candidate.citation_count,
             candidate.human_decision_count, candidate.package_hash, candidate.created_at),
        )
        self.conn.commit()
        return candidate.release_id

    def get_release_candidates(self, project_id: str) -> list[ReleaseCandidate]:
        rows = self.conn.execute(
            "SELECT * FROM release_candidates WHERE project_id = ? ORDER BY created_at", (project_id,)
        ).fetchall()
        return [
            ReleaseCandidate(
                release_id=row["release_id"], project_id=row["project_id"],
                manuscript_plan_id=row["manuscript_plan_id"], readiness_report_id=row["readiness_report_id"],
                completion_plan_id=row["completion_plan_id"], peer_review_round_id=row["peer_review_round_id"],
                section_count=row["section_count"], drafted_section_count=row["drafted_section_count"],
                citation_count=row["citation_count"], human_decision_count=row["human_decision_count"],
                package_hash=row["package_hash"], created_at=row["created_at"],
            )
            for row in rows
        ]

    def add_readiness_report(self, report: ReadinessReport) -> str:
        self.conn.execute(
            "INSERT INTO readiness_reports "
            "(report_id, project_id, run_id, readiness_status, audit_ids, finding_ids, "
            "blocking_finding_ids, explanation, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (report.report_id, report.project_id, report.run_id, report.readiness_status,
             json.dumps(report.audit_ids), json.dumps(report.finding_ids),
             json.dumps(report.blocking_finding_ids), report.explanation, report.created_at),
        )
        self.conn.commit()
        return report.report_id

    def get_readiness_report(self, project_id: str) -> ReadinessReport | None:
        row = self.conn.execute(
            "SELECT * FROM readiness_reports WHERE project_id = ? ORDER BY created_at DESC LIMIT 1", (project_id,)
        ).fetchone()
        if row is None:
            return None
        return ReadinessReport(
            report_id=row["report_id"], project_id=row["project_id"], run_id=row["run_id"],
            readiness_status=row["readiness_status"], audit_ids=json.loads(row["audit_ids"]),
            finding_ids=json.loads(row["finding_ids"]), blocking_finding_ids=json.loads(row["blocking_finding_ids"]),
            explanation=row["explanation"], created_at=row["created_at"],
        )

    def add_human_decision(self, decision: HumanDecision) -> str:
        self.conn.execute(
            "INSERT INTO human_decisions "
            "(decision_id, project_id, run_id, kind, decision, payload, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (decision.decision_id, decision.project_id, decision.run_id, decision.kind,
             decision.decision, json.dumps(decision.payload), decision.created_at),
        )
        self.conn.commit()
        return decision.decision_id

    def get_human_decisions(self, project_id: str) -> list[HumanDecision]:
        rows = self.conn.execute(
            "SELECT * FROM human_decisions WHERE project_id = ? ORDER BY created_at", (project_id,)
        ).fetchall()
        return [
            HumanDecision(
                decision_id=row["decision_id"], project_id=row["project_id"], run_id=row["run_id"],
                kind=row["kind"], decision=row["decision"], payload=json.loads(row["payload"]),
                created_at=row["created_at"],
            )
            for row in rows
        ]
