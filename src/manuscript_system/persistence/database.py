from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (
    project_id TEXT PRIMARY KEY,
    source_path TEXT NOT NULL,
    domain_profile_id TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS source_assets (
    artifact_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    relative_path TEXT NOT NULL,
    checksum_sha256 TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    media_type TEXT NOT NULL,
    modified_at TEXT
);

CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    thread_id TEXT NOT NULL,
    workflow_name TEXT NOT NULL,
    status TEXT NOT NULL,
    current_stage TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS findings (
    finding_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    severity TEXT NOT NULL,
    rule_id TEXT NOT NULL,
    message TEXT NOT NULL,
    blocking INTEGER NOT NULL DEFAULT 0,
    affected_claim_ids TEXT NOT NULL DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS claims (
    claim_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    text TEXT NOT NULL,
    claim_type TEXT NOT NULL,
    importance TEXT NOT NULL DEFAULT 'SUPPORTING',
    evidence_ids TEXT NOT NULL DEFAULT '[]',
    status TEXT NOT NULL DEFAULT 'CANDIDATE'
);

CREATE TABLE IF NOT EXISTS evidence_items (
    evidence_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    source_artifact_id TEXT NOT NULL,
    evidence_type TEXT NOT NULL,
    location TEXT NOT NULL,
    excerpt_or_value TEXT NOT NULL,
    extraction_confidence REAL NOT NULL DEFAULT 1.0
);

CREATE TABLE IF NOT EXISTS readiness_reports (
    report_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    readiness_status TEXT NOT NULL,
    audit_ids TEXT NOT NULL DEFAULT '[]',
    finding_ids TEXT NOT NULL DEFAULT '[]',
    blocking_finding_ids TEXT NOT NULL DEFAULT '[]',
    explanation TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS citations (
    citation_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    cite_key TEXT NOT NULL,
    bib_title TEXT NOT NULL DEFAULT '',
    bib_authors TEXT NOT NULL DEFAULT '',
    bib_year INTEGER,
    bib_doi TEXT,
    verification_status TEXT NOT NULL,
    verified_title TEXT,
    verified_doi TEXT,
    match_confidence REAL NOT NULL DEFAULT 0.0,
    notes TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS assessor_reports (
    report_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    assessor_id TEXT NOT NULL,
    scientific_contribution REAL NOT NULL DEFAULT 0.0,
    evidence_sufficiency REAL NOT NULL DEFAULT 0.0,
    methodological_rigor REAL NOT NULL DEFAULT 0.0,
    validation_strength REAL NOT NULL DEFAULT 0.0,
    reproducibility REAL NOT NULL DEFAULT 0.0,
    literature_positioning REAL NOT NULL DEFAULT 0.0,
    potential_significance REAL NOT NULL DEFAULT 0.0,
    major_risks TEXT NOT NULL DEFAULT '[]',
    contribution_candidates TEXT NOT NULL DEFAULT '[]',
    confidence REAL NOT NULL DEFAULT 0.0,
    abstain INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS claim_assessments (
    record_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    assessor_id TEXT NOT NULL,
    claim_id TEXT NOT NULL,
    label TEXT NOT NULL,
    score REAL NOT NULL DEFAULT 0.0,
    reasoning TEXT NOT NULL DEFAULT '',
    limitations TEXT NOT NULL DEFAULT '[]',
    missing_evidence TEXT NOT NULL DEFAULT '[]',
    confidence REAL NOT NULL DEFAULT 0.0
);

CREATE TABLE IF NOT EXISTS numeric_cross_checks (
    check_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    claimed_value REAL NOT NULL,
    claimed_text TEXT NOT NULL,
    source_section TEXT NOT NULL,
    status TEXT NOT NULL,
    matched_value REAL,
    matched_source_path TEXT,
    matched_context TEXT,
    precision_places INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS project_knowledge_maps (
    map_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    asset_counts_by_role TEXT NOT NULL DEFAULT '{}',
    total_assets INTEGER NOT NULL DEFAULT 0,
    total_size_bytes INTEGER NOT NULL DEFAULT 0,
    has_readme INTEGER NOT NULL DEFAULT 0,
    has_environment_file INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS completion_tasks (
    task_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    plan_id TEXT NOT NULL,
    title TEXT NOT NULL,
    reason TEXT NOT NULL,
    category TEXT NOT NULL,
    priority TEXT NOT NULL,
    affected_claim_ids TEXT NOT NULL DEFAULT '[]',
    source TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS completion_plans (
    plan_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    task_ids TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS planned_sections (
    section_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    plan_id TEXT NOT NULL,
    name TEXT NOT NULL,
    "order" INTEGER NOT NULL DEFAULT 0,
    claim_ids TEXT NOT NULL DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS manuscript_plans (
    plan_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    journal_id TEXT,
    section_ids TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS manuscript_blocks (
    block_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    plan_id TEXT NOT NULL,
    section_id TEXT NOT NULL,
    text TEXT NOT NULL,
    claim_ids TEXT NOT NULL DEFAULT '[]',
    evidence_ids TEXT NOT NULL DEFAULT '[]',
    literature_ids TEXT NOT NULL DEFAULT '[]',
    authoring_agent TEXT NOT NULL DEFAULT '',
    prompt_hash TEXT NOT NULL DEFAULT '',
    model_id TEXT NOT NULL DEFAULT '',
    code_revision TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'PROPOSED',
    semantic_warnings TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS peer_review_rounds (
    round_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    plan_id TEXT NOT NULL,
    thread_id TEXT NOT NULL,
    status TEXT NOT NULL,
    response_to_reviewers TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS human_decisions (
    decision_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    decision TEXT NOT NULL,
    payload TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);
"""


def connect(db_path: str) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn
