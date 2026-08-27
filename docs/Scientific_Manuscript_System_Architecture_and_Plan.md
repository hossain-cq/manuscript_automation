# Advanced Local Multi-Agent Scientific Manuscript System

## Architecture Review, Improvements, Diagrams, and Implementation Plan

**Prepared by:** Manus AI  
**Review basis:** The complete specification supplied in `pasted_content.txt`  
**Recommended architectural stance:** Local-first, evidence-centered, workflow-governed, human-supervised

## Executive assessment

The specification is unusually strong in its scientific-integrity goals. It correctly treats manuscript production as a controlled pipeline from research evidence to publication package rather than as a single text-generation prompt. It also identifies the most important failure modes: fabricated references, unsupported claims, silent modification of source data, inconsistent terminology, non-reproducible analysis, journal non-compliance, and overconfident interpretation.

The main adjustment required is architectural rather than conceptual. The current design is primarily an **agent catalog and linear pipeline**. To make it implementable and safe, it should become a **versioned evidence-and-artifact platform with a workflow control plane**. Agents should be replaceable workers that read immutable inputs and propose typed outputs. The orchestrator and policy engine should decide whether those outputs may advance. The manuscript should be assembled from approved, traceable blocks rather than edited as an unstructured shared document.

> **Central design rule:** Agents may propose claims, analyses, text, figures, and revisions; only governed workflow transitions may promote them into approved project artifacts.

## 1. What is already correct

The specification establishes the right priorities: scientific correctness over fluency, citation-backed claims, provenance, reproducibility, modular agents, local execution, version control, and mandatory human approval. The distinction between fact, result, interpretation, hypothesis, speculation, literature claim, and AI-generated suggestion is particularly important and should be represented as a first-class data model rather than only as a writing instruction.

The proposed specialized roles are also appropriate. Separating literature discovery from literature analysis, data inspection from scientific interpretation, section writers from validation agents, and journal formatting from journal compliance will reduce the risk that one model both generates and approves its own unsupported content.

The incremental development strategy is sound. The first demonstration should produce a human-reviewable manuscript and a readiness report, not attempt autonomous submission or autonomous scientific discovery.

## 2. Required architectural adjustments

| Area | Current direction | Recommended adjustment | Reason |
|---|---|---|---|
| Control | Master orchestrator coordinates agents | Add a workflow engine, policy engine, task queue, and agent registry | Prevents arbitrary agent sequencing and makes gates auditable |
| Data integrity | Immutable raw-data layer | Enforce content-addressed storage, checksums, read-only mounts, and derived-artifact lineage | Makes accidental or malicious mutation detectable |
| Manuscript representation | Draft manuscript plus versions | Represent claims, evidence links, manuscript blocks, and releases separately | Enables claim-level verification and selective revision |
| Communication | Structured JSON messages | Define versioned `TaskEnvelope` and `AgentResult` contracts validated by schemas | Eliminates free-form interoperability failures |
| Memory | Database, vector DB, document store, graph | Define deterministic system-of-record boundaries | Avoids treating embeddings as authoritative facts |
| Orchestration | Hierarchical agent architecture | Use a durable state machine with retries, idempotency, failure states, and human gates | Supports recovery and reproducibility |
| Models | Replaceable local model layer | Add a model gateway with capability routing, prompt/version hashes, and fallback policy | Prevents direct coupling between agents and model runtimes |
| External services | Metadata APIs and optional services | Place all network access behind allowlisted adapters with cached responses | Preserves local-first operation and records external evidence |
| Scientific execution | Domain tools listed as capabilities | Execute code and parsers in bounded scientific sandboxes | Controls filesystem, network, CPU, and dependency risk |
| Quality score | Multi-dimensional score | Treat it as a diagnostic dashboard with blocking rules separate from scores | Prevents a high average score from hiding a critical failure |
| Similarity | Similarity/plagiarism analysis | Report overlap categories and evidence; never infer misconduct automatically | Distinguishes common language from unattributed reuse |
| Journal support | Publisher and journal profiles | Use declarative versioned profiles with provenance and effective dates | Makes compliance rules maintainable and reviewable |

## 3. Target architecture

The target system has four planes. The **control plane** contains the API, workflow state machine, policy engine, task queue, agent registry, configuration, and human approval interface. The **agent execution plane** contains specialized workers. The **evidence and artifact plane** stores immutable inputs, structured project records, versioned outputs, claim/evidence/citation relationships, provenance events, and release snapshots. The **service adapter plane** exposes local models, scientific tools, document renderers, and optional external metadata providers through stable interfaces.

This separation addresses a central weakness in a pure multi-agent design: an agent should not be able to directly overwrite source data, alter a reference record without audit, or bypass a scientific review gate simply because it has access to a shared folder.

![Target layered architecture](https://private-us-east-1.manuscdn.com/sessionFile/OZka5VzJyJ9HJ9nlGpOrEo/sandbox/0HxMRVotiDGgy9pj1kIyZi-images_1787306801885_na1fn_L2hvbWUvdWJ1bnR1L21hbnVzY3JpcHRfYXJjaGl0ZWN0dXJlL3JlbmRlcmVkLzAxX3RhcmdldF9hcmNoaXRlY3R1cmU.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvT1prYTVWekp5SjlISjlubEdwT3JFby9zYW5kYm94LzBIeE1SVm90aURHZ3k5cGoxa0l5WmktaW1hZ2VzXzE3ODczMDY4MDE4ODVfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwyMWhiblZ6WTNKcGNIUmZZWEpqYUdsMFpXTjBkWEpsTDNKbGJtUmxjbVZrTHpBeFgzUmhjbWRsZEY5aGNtTm9hWFJsWTNSMWNtVS5wbmciLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3ODk0MzA0MDB9fX1dfQ__&Key-Pair-Id=K2QY5QTL8JSY6C&Signature=MEUCIGwcGl03Veut49P0J6xQ9ajpJ9A6RlAwlRCE3t4DpXSgAiEAlwkCztyRCAqe59Z1DJu2PufGvnniv4pMqnuyNo-fg6M_)

The editable source is [01_target_architecture.mmd](diagrams/01_target_architecture.mmd). The diagram shows the recommended boundary between orchestration, agents, evidence storage, and adapters.

## 4. Core domain model

The system should be designed around explicit domain entities. A project contains research questions, hypotheses, source collections, datasets, computational runs, literature records, claims, manuscript blocks, figures, tables, equations, journal profiles, review comments, decisions, and release candidates.

| Entity | Purpose | Must be versioned? | Key integrity fields |
|---|---|---:|---|
| `SourceAsset` | Original paper, dataset, output, image, or user file | Yes | checksum, origin, ingestion time, media type |
| `EvidenceItem` | Extracted passage, number, method, equation, or observation | Yes | source asset, location, extraction method, confidence |
| `Claim` | A scientific statement proposed for the manuscript | Yes | claim type, wording, evidence links, status, reviewer decisions |
| `AnalysisRun` | Reproducible computation or interpretation | Yes | code revision, environment lock, inputs, parameters, outputs |
| `ManuscriptBlock` | Paragraph, caption, equation, table, or section | Yes | block ID, claim IDs, citations, authoring agent, semantic diff |
| `JournalProfile` | Declarative target-journal rules | Yes | source, effective date, version, rule severity |
| `ValidationFinding` | Warning, failure, or pass from a validator | Yes | rule ID, artifact versions, severity, remediation |
| `HumanDecision` | Explicit approval, rejection, or override | Yes | actor, scope, rationale, timestamp |
| `ReleaseCandidate` | Immutable manuscript snapshot for QA or submission | Yes | included artifact IDs, report status, package hash |

The manuscript should not be stored only as a monolithic `.docx` or `.tex` file. Those formats remain important export targets, but the internal canonical representation should preserve block IDs, claims, citations, equations, figures, tables, and provenance links.

## 5. Evidence and provenance architecture

Every material statement should be mapped to one or more evidence items. Evidence can originate from user input, an experimental dataset, a computational result, verified literature, a figure, a table, or an agent inference. The system should store the distinction between direct evidence and interpretation, because a citation to a source does not automatically validate the exact wording of a newly generated conclusion.

![Evidence and provenance architecture](https://private-us-east-1.manuscdn.com/sessionFile/OZka5VzJyJ9HJ9nlGpOrEo/sandbox/0HxMRVotiDGgy9pj1kIyZi-images_1787306801885_na1fn_L2hvbWUvdWJ1bnR1L21hbnVzY3JpcHRfYXJjaGl0ZWN0dXJlL3JlbmRlcmVkLzAyX2V2aWRlbmNlX3Byb3ZlbmFuY2U.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvT1prYTVWekp5SjlISjlubEdwT3JFby9zYW5kYm94LzBIeE1SVm90aURHZ3k5cGoxa0l5WmktaW1hZ2VzXzE3ODczMDY4MDE4ODVfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwyMWhiblZ6WTNKcGNIUmZZWEpqYUdsMFpXTjBkWEpsTDNKbGJtUmxjbVZrTHpBeVgyVjJhV1JsYm1ObFgzQnliM1psYm1GdVkyVS5wbmciLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3ODk0MzA0MDB9fX1dfQ__&Key-Pair-Id=K2QY5QTL8JSY6C&Signature=MEUCIDH~VFgEX3af~aanj-E07u9G1f6nVnIJuBX4suMpliWRAiEA5Jt3TGt1XiuJrdCwHirQJ4MRPuoBApW9KcQg9GYDyFM_)

The recommended flow is ingestion, normalization, evidence extraction, claim registration, evidence linking, validation, human decision, and manuscript-block promotion. Each step produces a durable record. When a researcher asks, “Why did the manuscript say this?”, the system should answer with the claim ID, supporting evidence, source locations, transformations, model and code versions, validation findings, and approval history.

A provenance record should include at least:

```yaml
provenance:
  artifact_id: ART-0001
  parent_artifact_ids: []
  source_asset_ids: []
  claim_ids: []
  agent_id: scientific_analysis.v1
  model_id: local-model-or-rule-engine
  prompt_hash: sha256:...
  code_revision: git:...
  environment_lock_hash: sha256:...
  created_at: 2026-08-21T00:00:00Z
  human_decision_ids: []
```

## 6. Workflow and human governance

The manuscript lifecycle should be a durable state machine, not merely a sequence written in documentation. Each transition must have entry criteria, exit criteria, permitted actors, blocking findings, and an explicit override path. Critical stages must be impossible to skip through ordinary agent calls.

![Governed manuscript workflow](https://private-us-east-1.manuscdn.com/sessionFile/OZka5VzJyJ9HJ9nlGpOrEo/sandbox/0HxMRVotiDGgy9pj1kIyZi-images_1787306801885_na1fn_L2hvbWUvdWJ1bnR1L21hbnVzY3JpcHRfYXJjaGl0ZWN0dXJlL3JlbmRlcmVkLzAzX3dvcmtmbG93X3N0YXRl.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvT1prYTVWekp5SjlISjlubEdwT3JFby9zYW5kYm94LzBIeE1SVm90aURHZ3k5cGoxa0l5WmktaW1hZ2VzXzE3ODczMDY4MDE4ODVfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwyMWhiblZ6WTNKcGNIUmZZWEpqYUdsMFpXTjBkWEpsTDNKbGJtUmxjbVZrTHpBelgzZHZjbXRtYkc5M1gzTjBZWFJsLnBuZyIsIkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTc4OTQzMDQwMH19fV19&Key-Pair-Id=K2QY5QTL8JSY6C&Signature=MEQCIB~WxntW7cUJhnK1bYdxxg2uw9uz-sn0AGMykt9DV9BDAiA9lsqFFu-p3uj3XFZWSNM7wu28P70t2zVHl7s1677tOQ__)

Mandatory gates should occur after source/data ingestion, after the manuscript plan, before final scientific conclusions are accepted, when uncertain references are proposed, after semantic language edits, before the final response to reviewers, and before package generation. Submission itself must remain a manual action. The system may generate a submission package, but it must not submit it automatically.

The policy engine should distinguish **blocking rules** from **diagnostic scores**. For example, an unverifiable DOI, a contradiction between a figure and a central numerical claim, or an unreviewed new experimental claim should block readiness regardless of the overall quality score.

## 7. Agent organization

The requested agent list should be retained, but implemented as capability-oriented workers behind stable interfaces. Several roles can initially share one process and one environment. The logical separation is more important than creating a separate operating-system process or Conda environment for every role.

| Capability group | Initial workers | Primary inputs | Primary outputs |
|---|---|---|---|
| Project and control | Project Manager, Orchestrator | project manifest, tasks, decisions | plans, task envelopes, state transitions |
| Literature | Discovery, Analysis, Citation, Bibliography | queries, PDFs, metadata | verified references, literature matrix, citation findings |
| Data and science | Data/Results, Scientific Analysis, Domain Parsers | immutable source assets, code, parameters | normalized results, analysis runs, candidate claims |
| Planning and writing | Planner, Introduction, Methods, Results, Discussion, Conclusion, Abstract | approved claims, journal profile, terminology | manuscript blocks, outline, captions |
| Scientific QA | Fact Checker, Equation Checker, Consistency, Reproducibility | manuscript blocks, evidence graph, analysis runs | validation findings, confidence, remediation tasks |
| Publication QA | Grammar, Proofreading, Similarity, Journal Compliance | release candidate, journal profile | publication reports and corrected proposals |
| Review cycle | Reviewer personas, Revision, Response-to-Reviewers | draft, evidence, journal context | review comments, revision plan, response document |
| Packaging | Assembler, Package Generator | approved blocks and reports | LaTeX/DOCX/PDF, supplement, cover letter, checklist |

Writers should not invent missing facts. If an approved claim or method parameter is unavailable, the writer should emit a structured `MISSING_EVIDENCE` finding and leave a controlled placeholder or request human input.

## 8. Agent contract

All workers should accept a versioned task envelope and return a versioned result envelope. The envelope must carry artifact references rather than embedding large documents or silently passing mutable state.

![Structured agent execution contract](https://private-us-east-1.manuscdn.com/sessionFile/OZka5VzJyJ9HJ9nlGpOrEo/sandbox/0HxMRVotiDGgy9pj1kIyZi-images_1787306801885_na1fn_L2hvbWUvdWJ1bnR1L21hbnVzY3JpcHRfYXJjaGl0ZWN0dXJlL3JlbmRlcmVkLzA1X2FnZW50X2NvbnRyYWN0.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvT1prYTVWekp5SjlISjlubEdwT3JFby9zYW5kYm94LzBIeE1SVm90aURHZ3k5cGoxa0l5WmktaW1hZ2VzXzE3ODczMDY4MDE4ODVfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwyMWhiblZ6WTNKcGNIUmZZWEpqYUdsMFpXTjBkWEpsTDNKbGJtUmxjbVZrTHpBMVgyRm5aVzUwWDJOdmJuUnlZV04wLnBuZyIsIkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTc4OTQzMDQwMH19fV19&Key-Pair-Id=K2QY5QTL8JSY6C&Signature=MEUCIDf-m6CWuGIawbWod-rcV2QhCNE0PUCqsWS60Vc~53xSAiEA2aa3EOJiejWV9zSYmGIsdeExEM7ZW9pKWjorHajCHyw_)

A minimal contract should include `task_id`, `agent_id`, `capability`, `input_artifact_ids`, `constraints`, `schema_version`, `output_artifact_ids`, `evidence`, `confidence`, `warnings`, `status`, `requires_human_review`, and provenance metadata. Valid statuses should include `SUCCEEDED`, `FAILED`, `BLOCKED`, and `NEEDS_REVIEW`.

The contract should also define idempotency. Re-running the same task with identical input artifact IDs, configuration, model identity, code revision, and prompt hash should either return the same cached result or produce a clearly different run ID. Silent non-deterministic replacement of artifacts should not be allowed.

## 9. Storage and memory strategy

The proposed combination of structured database, document store, vector index, citation graph, and metadata store is appropriate if each has a clear role. PostgreSQL or SQLite can hold project state, structured metadata, findings, decisions, and normalized records. A content-addressed filesystem or object store can hold raw and derived files. A vector index can accelerate semantic retrieval but must never be the authoritative source for numerical values, DOI metadata, workflow state, or approval status. A graph representation can connect claims, evidence, references, figures, tables, equations, and manuscript blocks.

![Local deployment and repository architecture](https://private-us-east-1.manuscdn.com/sessionFile/OZka5VzJyJ9HJ9nlGpOrEo/sandbox/0HxMRVotiDGgy9pj1kIyZi-images_1787306801885_na1fn_L2hvbWUvdWJ1bnR1L21hbnVzY3JpcHRfYXJjaGl0ZWN0dXJlL3JlbmRlcmVkLzA0X2RlcGxveW1lbnRfcmVwb3NpdG9yeQ.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvT1prYTVWekp5SjlISjlubEdwT3JFby9zYW5kYm94LzBIeE1SVm90aURHZ3k5cGoxa0l5WmktaW1hZ2VzXzE3ODczMDY4MDE4ODVfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwyMWhiblZ6WTNKcGNIUmZZWEpqYUdsMFpXTjBkWEpsTDNKbGJtUmxjbVZrTHpBMFgyUmxjR3h2ZVcxbGJuUmZjbVZ3YjNOcGRHOXllUS5wbmciLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3ODk0MzA0MDB9fX1dfQ__&Key-Pair-Id=K2QY5QTL8JSY6C&Signature=MEUCIDj7iZyFLQ20WwDZqguqSC-ImJv4nwcJVCq5n5AQcVccAiEA-ZcoYPo-1BUG6HT~NOeSsAVMCh7K0DeXvCle1H3gP7w_)

A recommended project layout is:

```text
scientific-multi-agent/
├── apps/                         # API, CLI, local UI
├── domain/                       # typed entities and business rules
├── agents/                       # capability workers and adapters
├── workflows/                    # state machines and transition policies
├── validators/                   # scientific, citation, journal, language QA
├── storage/                      # DB, artifact store, vector and graph adapters
├── model_gateway/                # local model routing and inference contracts
├── scientific_tools/             # bounded domain parsers and runners
├── journals/                     # versioned declarative profiles
├── schemas/                      # JSON Schema / Pydantic contracts
├── prompts/                      # versioned prompt templates
├── projects/<project_id>/
│   ├── raw/                      # immutable, checksum-verified inputs
│   ├── derived/                  # reproducible transformed artifacts
│   ├── literature/
│   ├── manuscript/
│   ├── figures/
│   ├── tables/
│   ├── reports/
│   └── releases/
├── tests/                        # unit, integration, golden and integrity tests
├── environments/                 # grouped Conda lockfiles
└── docs/
```

## 10. Conda and execution boundaries

Do not create one environment per logical agent by default. Start with four or five dependency domains and split only when conflicts or security requirements justify it.

| Environment | Contents | Typical users |
|---|---|---|
| `manuscript-core` | API, orchestration, Pydantic, workflow, storage clients | orchestrator, project manager, planner |
| `agent-language-qa` | LLM client, retrieval, text processing, citation logic | writers, literature, citation, review, language QA |
| `scientific-tools` | NumPy, SciPy, pandas, domain parsers, ASE/pymatgen and selected tools | data, analysis, equation and reproducibility agents |
| `local-llm` | model runtime, tokenizer, quantization and embedding services | model gateway |
| `document-rendering` | LaTeX, Pandoc, PDF parsing, figure/table rendering | assembler, journal and package agents |

Heavy domain software such as DFT or quantum-chemistry packages should be optional profiles or separate tool environments. Each analysis run should record the exact environment lockfile hash, executable versions, command-line parameters, input checksums, and output checksums.

## 11. Journal-profile architecture

Journal compliance should be declarative. A profile should contain manuscript structure, limits, required statements, citation rules, bibliography rules, figure/table rules, supplementary-information rules, and package requirements. Profiles must be versioned and should record the source and effective date of each rule. Publisher-level defaults can be inherited by journal profiles and overridden only with explicit metadata.

The compliance engine should output `PASS`, `WARNING`, or `FAIL` for each rule and should distinguish “not verifiable automatically” from “does not comply.” The system should never present a stale or incomplete journal profile as authoritative without warning the user.

## 12. Implementation roadmap

### Phase 0 — Architecture baseline and golden project

Select one existing computational materials, chemistry, energy-materials, or quantum project as the demonstration case. Freeze a small, representative input set containing papers, raw outputs, figures, tables, code, and the intended target journal. Define success criteria around traceability and human review rather than prose quality alone.

**Exit criteria:** a project manifest exists; all inputs have checksums; the target journal is identified; at least ten representative scientific claims are registered manually for testing.

### Phase 1 — Core infrastructure

Implement the project manifest, immutable artifact store, structured database, schema validation, audit logging, configuration, task envelopes, agent registry, and a durable workflow state machine. Build the local API and CLI first. Add the human approval inbox before adding many agents.

**Exit criteria:** a project can be created, assets ingested, artifacts versioned, tasks executed, failures recorded, and transitions approved or blocked without any LLM dependency.

### Phase 2 — Evidence and research intelligence

Implement literature metadata adapters, PDF ingestion, reference normalization, literature matrices, dataset inspection, domain parsers, and analysis-run records. Add the claim registry and claim–evidence–citation graph. Keep retrieval deterministic: exact IDs, source locations, and checksums must be available alongside semantic search.

**Exit criteria:** the system can answer which source supports each registered claim and can refuse to accept an unverifiable reference.

### Phase 3 — Manuscript planning and controlled generation

Implement the manuscript planner, journal-profile loader, terminology dictionary, block-based manuscript model, and section writers. Writers should consume approved claims and evidence, cite only verified references, and emit missing-evidence findings instead of filling gaps with invented content. Generate a first human-reviewable manuscript candidate.

**Exit criteria:** every material generated statement is linked to a claim or marked as an explicitly labeled suggestion; the author can approve, reject, and revise individual blocks.

### Phase 4 — Scientific quality assurance

Implement fact checking, numerical consistency, equation checks, units and terminology checks, figure/table cross-reference checks, reproducibility checks, and scientific peer-review personas. Add blocking thresholds and remediation tasks. Preserve semantic diffs for language edits so numerical and scientific changes are visible.

**Exit criteria:** seeded test cases containing fabricated DOI data, contradictory numbers, unit errors, orphaned figures, and unsupported claims are detected with actionable findings.

### Phase 5 — Publication quality assurance

Implement bibliography formatting, journal compliance, grammar, proofreading, similarity classification, document rendering, and readiness reporting. Keep formatting profiles separate from scientific content. Generate LaTeX and DOCX where practical, plus a PDF only as an export artifact.

**Exit criteria:** the system generates a journal-specific compliance report and an immutable release candidate with all required package components.

### Phase 6 — Review and revision automation

Implement reviewer-comment ingestion, comment categorization, revision proposals, response-to-reviewers generation, and change tracking. Every response should link to the addressed manuscript block, evidence, reviewer comment, and human decision.

**Exit criteria:** a complete simulated revision cycle produces an auditable response document without automatically accepting invalid reviewer requests.

### Phase 7 — Hardening and domain expansion

Add performance profiling, offline operation tests, backup/restore, model replacement tests, permission boundaries, provenance export, and domain plug-ins for additional disciplines. Introduce optional external services only through adapters with explicit configuration and cached results.

**Exit criteria:** the system can be rebuilt from repository revision, environment locks, project artifacts, and configuration, and can reproduce the demonstration release reports.

## 13. Quality and acceptance strategy

The system should be tested with deliberately adversarial fixtures rather than only successful examples. Golden projects should include correct and incorrect citations, modified source files, inconsistent units, mismatched figure values, ambiguous interpretations, duplicate references, incomplete methods, and journal-profile violations.

| Test category | Example acceptance test |
|---|---|
| Integrity | A raw file modification changes its checksum and blocks downstream release |
| Citation | A fabricated DOI is never promoted to a verified reference |
| Claims | A major claim without evidence creates a blocking finding |
| Numbers | A manuscript value inconsistent with the source result is detected |
| Provenance | Every release paragraph can be traced to claims and source artifacts |
| Workflow | Critical validation stages cannot be skipped without a recorded override |
| Human control | Final conclusions and submission package require explicit approval |
| Reproducibility | An analysis run records code, parameters, environment, inputs, and outputs |
| Formatting | A journal profile produces deterministic compliance findings |
| Recovery | Failed agents retry safely without duplicating or corrupting artifacts |

## 14. Priority risks and mitigations

| Risk | Severity | Mitigation |
|---|---:|---|
| Hallucinated scientific content | Critical | claim registry, evidence requirements, blocking policy, human gates |
| Silent mutation of source data | Critical | immutable raw store, checksums, read-only execution mounts |
| False confidence from aggregate scores | High | blocking rules independent of quality scores |
| Model or prompt drift | High | model gateway, prompt hashes, versioned evaluation sets |
| Unverifiable external metadata | High | allowlisted adapters, cached responses, source timestamps, explicit unknown status |
| Agent dependency conflicts | Medium | grouped Conda environments and lockfiles |
| Workflow deadlocks | Medium | explicit blocked state, retry policy, timeout, manual recovery actions |
| Overly complex first release | High | build one end-to-end golden project before broad agent coverage |
| Similarity misclassification | Medium | classify overlap and show evidence; do not label plagiarism automatically |
| Stale journal rules | Medium | profile version, source, effective date, and warning state |

## 15. Recommended first vertical slice

The first useful release should not implement every requested agent. It should implement one complete, trustworthy path:

```text
Create project
  → ingest papers, datasets, figures, and code
  → verify checksums and metadata
  → build literature matrix
  → inspect one results dataset
  → register and evidence-link claims
  → create journal-aware outline
  → generate Methods and Results blocks
  → run citation, numerical, terminology, and reproducibility checks
  → obtain human approval
  → render a reviewable manuscript and readiness report
```

This slice tests the highest-value architectural properties: evidence traceability, immutable data handling, structured agent communication, workflow gates, and reproducible outputs. Introduction, Discussion, reviewer simulation, similarity analysis, and automated package generation can be added after this foundation is stable.

## 16. Final recommendation

Proceed with the project, but revise the implementation framing from “many agents coordinated by a master agent” to **“a governed scientific evidence platform whose workers happen to be agents.”** Keep the requested roles as capabilities, not as uncontrolled autonomous actors. Make claims, evidence, artifacts, validation findings, and human decisions first-class records. Use the orchestrator to manage state and dependencies, the policy engine to enforce integrity, and the artifact/provenance layer to make every result inspectable.

With these changes, the system can remain modular, local-first, domain-aware, and extensible without sacrificing scientific conservatism. The architecture is suitable for an initial computational materials and quantum-science implementation while retaining clean extension points for chemistry, biology, engineering, and other research domains.

## Diagram files

| Diagram | Editable source | Rendered preview |
|---|---|---|
| Target layered architecture | [01_target_architecture.mmd](diagrams/01_target_architecture.mmd) | [01_target_architecture.png](rendered/01_target_architecture.png) |
| Evidence and provenance | [02_evidence_provenance.mmd](diagrams/02_evidence_provenance.mmd) | [02_evidence_provenance.png](rendered/02_evidence_provenance.png) |
| Workflow state machine | [03_workflow_state.mmd](diagrams/03_workflow_state.mmd) | [03_workflow_state.png](rendered/03_workflow_state.png) |
| Deployment and repository | [04_deployment_repository.mmd](diagrams/04_deployment_repository.mmd) | [04_deployment_repository.png](rendered/04_deployment_repository.png) |
| Agent execution contract | [05_agent_contract.mmd](diagrams/05_agent_contract.mmd) | [05_agent_contract.png](rendered/05_agent_contract.png) |

## Closing note

The supplied specification is a strong requirements document and a good basis for implementation. Its most important improvement is to formalize the boundaries between **evidence, claims, manuscript text, validation, and approval**. Those boundaries are now reflected in the target diagrams and roadmap above.
