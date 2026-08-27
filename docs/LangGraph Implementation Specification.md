# LangGraph Implementation Specification

## Local, evidence-centered, multi-agent scientific manuscript system

**Status:** Implementation baseline  
**Primary orchestration runtime:** LangGraph  
**Recommended first domains:** DFT/computational materials, AI/ML, quantum computing and quantum chemistry  
**Execution model:** Local-first, single-user, human-supervised, versioned, auditable

## 1. Purpose and final architectural decision

This document converts the approved scientific manuscript-system concept into an implementation specification centered on LangGraph. It defines how the system should accept a local project folder, understand the research, audit scientific readiness, create a claim–evidence map, plan a manuscript, generate controlled manuscript blocks, run scientific and publication QA, and produce a human-approved release package.

LangGraph is an appropriate core orchestration runtime because its model is based on typed graph state, nodes, and edges, and because it supports durable state, persistence, streaming, human-in-the-loop interrupts, and deterministic steps mixed with LLM-driven steps [1] [2] [3] [4]. It should not, however, be treated as the complete application. LangGraph should coordinate workflow state and agent execution while the application’s structured database, artifact store, provenance ledger, policy engine, and scientific job runner remain separate components.

> **Final decision:** Use LangGraph as the workflow and agent-orchestration layer, not as the authoritative scientific database, artifact store, or unrestricted code-execution environment.

The first implementation should remain a modular Python application. It should begin with a local CLI and a small API, use SQLite and a content-addressed filesystem for development, and later support PostgreSQL and more durable worker execution when the workload grows.

## 2. System objective

The system must transform a completed research project into a defensible manuscript-development package through the following controlled process:

```text
Local project folder
  → read-only intake
  → deterministic discovery
  → project knowledge map
  → domain classification
  → workflow reconstruction
  → scientific audit
  → claim–evidence graph
  → publication-readiness assessment
  → human approval
  → manuscript plan
  → evidence-linked manuscript blocks
  → scientific QA
  → journal and publication QA
  → human-approved release package
```

The system must answer the following questions before writing a full manuscript:

| Question | Required system response |
|---|---|
| What did the researcher actually do? | Reconstructed workflow with source artifacts, code, parameters, and unresolved links |
| What evidence exists? | Evidence inventory connected to raw files, outputs, figures, tables, and literature |
| What may be claimed? | Candidate claims classified as facts, results, interpretations, hypotheses, literature claims, or suggestions |
| Is the evidence sufficient? | Claim–evidence gap report with missing controls, baselines, uncertainty, or validation |
| Is the work manuscript-ready? | Status such as ready, draftable with warnings, needs additional analysis, or insufficient evidence |
| What should be done next? | Human-reviewable research-completion plan or manuscript plan |
| Can the manuscript be audited? | Provenance from manuscript block to claim, evidence, artifact, computation, model, and approval |

The system must never guarantee novelty, publication, journal acceptance, or scientific truth. It provides a structured assessment with explicit uncertainty and human approval.

## 3. Why LangGraph, and what it should not do

LangGraph models workflows as graphs composed of **state**, **nodes**, and **edges**. Nodes may contain deterministic Python code, calls to scientific tools, or LLM-driven reasoning. Edges determine sequencing, branching, looping, and termination [4]. This is a good fit for the project because the workflow contains deterministic discovery steps, parallel audits, conditional research-completion loops, human approval gates, and iterative manuscript revision.

LangGraph persistence provides thread-scoped checkpoints, while stores provide application-defined data across threads [2]. This distinction is important for this system. A LangGraph checkpoint should preserve the current workflow execution, pending approval, and transient coordination state. The authoritative project knowledge, evidence, claims, artifacts, findings, and release manifests should live in the application database and artifact store.

LangGraph interrupts are appropriate for approval gates because they pause execution, persist state, and resume later using the same thread identifier [3]. Any side effect before an interrupt must be idempotent or moved after the interrupt, because the interrupted node may restart from its beginning when resumed [3].

| Responsibility | LangGraph | External application component |
|---|---|---|
| Workflow state and routing | Yes | No |
| Agent/subgraph orchestration | Yes | No |
| Human approval pauses | Yes, using interrupts | UI/API presents and records decisions |
| Thread checkpoints | Yes, persistent checkpointer | Database backend used by checkpointer |
| Project metadata | No authoritative ownership | Structured database |
| Raw and derived files | No | Content-addressed artifact store |
| Scientific provenance | No authoritative ownership | Provenance tables and append-only events |
| Arbitrary code execution | No | Bounded scientific job runner |
| Model routing | Via node/tool adapters | Model gateway |
| Journal rules | No | Versioned profile registry |
| Observability | Graph events and optional tracing | Logs, metrics, evaluation datasets |

## 4. Operating modes

The application should expose two top-level LangGraph workflows.

### 4.1 Project Assessment Graph

This graph begins with a local folder and ends with a Project Publication Readiness Report, a claim–evidence matrix, and either a manuscript-plan recommendation or a research-completion plan.

```text
intake_request
  → validate_boundary
  → create_source_manifest
  → discover_artifacts
  → classify_domain
  → build_knowledge_map
  → reconstruct_workflow
  → run_parallel_audits
  → synthesize_readiness
  → human_review_assessment
  → route_to_manuscript_or_completion
```

### 4.2 Manuscript Production Graph

This graph accepts an approved assessment and generates a controlled manuscript release candidate.

```text
approved_assessment
  → select_journal_profile
  → build_manuscript_plan
  → register_claims
  → generate_section_blocks
  → generate_figures_tables_captions
  → citation_audit
  → scientific_qa
  → human_review_draft
  → revise_or_continue
  → language_and_proof_qa
  → journal_compliance
  → similarity_classification
  → final_readiness
  → human_release_approval
  → package_release
```

These graphs should be separate compiled graphs or a parent graph with clearly bounded subgraphs. Separate top-level graphs make it harder for a writing path to bypass the assessment path and allow a researcher to re-run manuscript production from a previously approved assessment.

## 5. LangGraph graph hierarchy

The recommended hierarchy is:

```text
application_graph
├── assessment_graph
│   ├── discovery_subgraph
│   ├── knowledge_map_subgraph
│   ├── domain_classification_subgraph
│   ├── workflow_reconstruction_subgraph
│   ├── novelty_audit_subgraph
│   ├── evidence_audit_subgraph
│   ├── methodology_audit_subgraph
│   ├── consistency_audit_subgraph
│   ├── readiness_synthesis_subgraph
│   └── assessment_approval_subgraph
└── manuscript_graph
    ├── journal_selection_subgraph
    ├── manuscript_planning_subgraph
    ├── claim_registration_subgraph
    ├── section_generation_subgraph
    ├── figure_table_subgraph
    ├── citation_subgraph
    ├── scientific_qa_subgraph
    ├── revision_subgraph
    ├── publication_qa_subgraph
    ├── readiness_subgraph
    └── release_approval_subgraph
```

Each subgraph should have an explicit input and output contract. Do not rely on a large untyped global state containing the entire project. LangGraph supports input, output, and private schemas, but broad streaming can expose private channels, so the application should explicitly filter streamed outputs and keep sensitive details in the external persistence layer [4].

## 6. Repository layout

```text
scientific_manuscript_system/
├── pyproject.toml
├── README.md
├── Makefile
├── .env.example
├── configs/
│   ├── app.yaml
│   ├── policies.yaml
│   ├── models.yaml
│   └── domains/
│       ├── dft_materials.yaml
│       ├── ai_ml.yaml
│       └── quantum.yaml
├── src/manuscript_system/
│   ├── main.py
│   ├── settings.py
│   ├── api/
│   │   ├── app.py
│   │   ├── routes_projects.py
│   │   ├── routes_runs.py
│   │   ├── routes_approvals.py
│   │   └── routes_artifacts.py
│   ├── domain/
│   │   ├── models.py
│   │   ├── enums.py
│   │   ├── claims.py
│   │   ├── evidence.py
│   │   ├── artifacts.py
│   │   └── release.py
│   ├── persistence/
│   │   ├── database.py
│   │   ├── repositories.py
│   │   ├── artifact_store.py
│   │   ├── provenance.py
│   │   └── checkpointer.py
│   ├── graphs/
│   │   ├── assessment.py
│   │   ├── manuscript.py
│   │   ├── common_state.py
│   │   ├── routing.py
│   │   └── subgraphs/
│   │       ├── discovery.py
│   │       ├── audits.py
│   │       ├── literature.py
│   │       ├── writing.py
│   │       └── publication_qa.py
│   ├── nodes/
│   │   ├── intake.py
│   │   ├── discovery.py
│   │   ├── classification.py
│   │   ├── audits.py
│   │   ├── approvals.py
│   │   ├── manuscript.py
│   │   └── release.py
│   ├── agents/
│   │   ├── base.py
│   │   ├── literature.py
│   │   ├── scientific_analysis.py
│   │   ├── writers.py
│   │   ├── reviewers.py
│   │   └── validators.py
│   ├── tools/
│   │   ├── filesystem.py
│   │   ├── parsers/
│   │   ├── scientific_jobs.py
│   │   ├── literature_adapters.py
│   │   ├── rendering.py
│   │   └── model_gateway.py
│   ├── policies/
│   │   ├── integrity.py
│   │   ├── transitions.py
│   │   └── permissions.py
│   ├── schemas/
│   │   ├── graph_state.py
│   │   ├── messages.py
│   │   ├── reports.py
│   │   └── journal_profiles.py
│   └── observability/
│       ├── logging.py
│       ├── events.py
│       └── evaluation.py
├── journals/
├── prompts/
├── migrations/
├── tests/
│   ├── unit/
│   ├── graph/
│   ├── integrity/
│   ├── fixtures/
│   └── golden_projects/
├── environments/
│   ├── manuscript-core.yaml
│   ├── discovery-analysis.yaml
│   ├── agent-language.yaml
│   ├── local-llm.yaml
│   └── document-rendering.yaml
└── projects/
    └── managed/<project_id>/
        ├── manifest.json
        ├── raw_registry/
        ├── derived_artifacts/
        ├── knowledge_map/
        ├── audits/
        ├── manuscript/
        ├── reports/
        └── releases/
```

## 7. State design

Do not place entire files, full PDFs, large arrays, or complete manuscript documents into LangGraph state. State should contain IDs, compact summaries, routing decisions, status flags, and references to external artifacts.

### 7.1 Shared graph context

```python
from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal, TypedDict
import operator

RunStatus = Literal[
    "CREATED",
    "RUNNING",
    "WAITING_FOR_HUMAN",
    "SUCCEEDED",
    "FAILED",
    "BLOCKED",
    "CANCELLED",
]

ReadinessStatus = Literal[
    "READY_FOR_MANUSCRIPT",
    "DRAFTABLE_WITH_WARNINGS",
    "NEEDS_ADDITIONAL_ANALYSIS",
    "NEEDS_RESEARCH_COMPLETION",
    "INSUFFICIENT_EVIDENCE",
    "BLOCKED",
]

class RunContext(TypedDict, total=False):
    project_id: str
    run_id: str
    thread_id: str
    workflow_name: str
    workflow_version: str
    status: RunStatus
    created_at: str
    updated_at: str
    current_stage: str
    target_journal_id: str | None
    domain_id: str | None
    approved_by: str | None
    error_ids: list[str]
```

### 7.2 Assessment state

```python
class AssessmentState(TypedDict, total=False):
    context: RunContext
    source_path: str
    source_manifest_id: str
    source_artifact_ids: list[str]
    exclusion_rules: list[str]
    execution_authorized: bool
    discovery_report_id: str
    knowledge_map_id: str
    workflow_map_id: str
    domain_profile_id: str
    domain_confidence: float
    audit_ids: Annotated[list[str], operator.add]
    finding_ids: Annotated[list[str], operator.add]
    candidate_claim_ids: Annotated[list[str], operator.add]
    readiness_report_id: str | None
    readiness_status: ReadinessStatus | None
    completion_plan_id: str | None
    assessment_approval_id: str | None
    human_decision: dict | None
```

### 7.3 Manuscript state

```python
class ManuscriptState(TypedDict, total=False):
    context: RunContext
    assessment_id: str
    target_journal_id: str
    journal_profile_version: str
    manuscript_plan_id: str
    claim_ids: list[str]
    manuscript_block_ids: Annotated[list[str], operator.add]
    figure_ids: Annotated[list[str], operator.add]
    table_ids: Annotated[list[str], operator.add]
    citation_audit_id: str | None
    scientific_qa_id: str | None
    publication_qa_id: str | None
    reviewer_report_ids: Annotated[list[str], operator.add]
    revision_plan_id: str | None
    readiness_report_id: str | None
    release_candidate_id: str | None
    blocking_finding_ids: Annotated[list[str], operator.add]
    approval_id: str | None
    next_action: str | None
```

The state should use reducers only where parallel branches must accumulate independent IDs or findings. Scalar decisions such as `readiness_status`, `next_action`, and `approval_id` should use replacement semantics and be written by one controlled node. This prevents accidental nondeterministic merging.

## 8. External domain models

LangGraph state is not the source of truth. The application database should store the following records.

| Record | Purpose |
|---|---|
| `Project` | Project identity, owner, domain, source path, configuration, and lifecycle |
| `SourceAsset` | Immutable file registration with checksum, size, type, path, and ingestion status |
| `DerivedArtifact` | Generated report, extracted table, plot, parsed metadata, or document with parent IDs |
| `KnowledgeMap` | Structured representation of project methods, inputs, runs, results, and links |
| `EvidenceItem` | A specific passage, number, output, figure region, table cell, equation, or user statement |
| `Claim` | Candidate or approved scientific statement with type and evidence links |
| `Audit` | Novelty, evidence, methodology, consistency, reproducibility, or domain audit result |
| `Finding` | Severity, rule, message, evidence, affected artifacts, and remediation |
| `HumanDecision` | Approval, rejection, correction, or override with identity and rationale |
| `ManuscriptBlock` | Paragraph, equation, caption, table, or section with claim and citation links |
| `JournalProfile` | Versioned, declarative target-journal requirements |
| `ReleaseCandidate` | Immutable package manifest with all artifact and report versions |

## 9. Artifact and file safety model

The user’s project path must be validated before scanning. The system should reject paths that are not directories, are inaccessible, traverse outside permitted roots, or violate configured exclusions. It should record symbolic-link behavior explicitly and avoid following links outside the allowed project boundary by default.

Raw files should be registered with a cryptographic checksum, size, modification time, and canonical path. The managed workspace should store a source registry and, where configured, a read-only copy or content-addressed representation. Derived files must have parent artifact IDs and a generation record.

No graph node should write to the original project directory. Scientific jobs should receive read-only mounts of selected inputs and a separate writable output directory. The job runner should enforce CPU, memory, wall-clock, process, network, and storage limits.

## 10. Graph construction patterns

### 10.1 Common graph factory

```python
from langgraph.graph import StateGraph, START, END


def build_assessment_graph(checkpointer, store):
    builder = StateGraph(AssessmentState)
    builder.add_node("validate_boundary", validate_boundary)
    builder.add_node("create_manifest", create_manifest)
    builder.add_node("discover", discover)
    builder.add_node("classify_domain", classify_domain)
    builder.add_node("build_knowledge_map", build_knowledge_map)
    builder.add_node("reconstruct_workflow", reconstruct_workflow)
    builder.add_node("run_audits", run_audits)
    builder.add_node("synthesize_readiness", synthesize_readiness)
    builder.add_node("human_assessment_review", human_assessment_review)
    builder.add_node("route_after_assessment", route_after_assessment)
    builder.add_node("create_completion_plan", create_completion_plan)
    builder.add_node("prepare_manuscript_handoff", prepare_manuscript_handoff)

    builder.add_edge(START, "validate_boundary")
    builder.add_edge("validate_boundary", "create_manifest")
    builder.add_edge("create_manifest", "discover")
    builder.add_edge("discover", "classify_domain")
    builder.add_edge("classify_domain", "build_knowledge_map")
    builder.add_edge("build_knowledge_map", "reconstruct_workflow")
    builder.add_edge("reconstruct_workflow", "run_audits")
    builder.add_edge("run_audits", "synthesize_readiness")
    builder.add_edge("synthesize_readiness", "human_assessment_review")
    builder.add_conditional_edges(
        "human_assessment_review",
        route_after_assessment,
        {
            "manuscript": "prepare_manuscript_handoff",
            "completion": "create_completion_plan",
            "blocked": END,
        },
    )
    builder.add_edge("prepare_manuscript_handoff", END)
    builder.add_edge("create_completion_plan", END)

    return builder.compile(checkpointer=checkpointer, store=store)
```

The exact LangGraph API may evolve, so implementation should pin compatible package versions and run graph-construction tests in CI. The important architectural rule is that graph construction remains declarative and that node side effects are isolated behind application services.

### 10.2 Parallel audit pattern

Novelty, evidence, methodology, consistency, and reproducibility audits should run as independent branches after the knowledge map and workflow reconstruction are available. They should return audit IDs and finding IDs rather than mutating the same scalar fields.

```python
builder.add_node("novelty_audit", novelty_audit)
builder.add_node("evidence_audit", evidence_audit)
builder.add_node("methodology_audit", methodology_audit)
builder.add_node("consistency_audit", consistency_audit)
builder.add_node("reproducibility_audit", reproducibility_audit)
builder.add_node("synthesize_readiness", synthesize_readiness)

builder.add_edge("reconstruct_workflow", "novelty_audit")
builder.add_edge("reconstruct_workflow", "evidence_audit")
builder.add_edge("reconstruct_workflow", "methodology_audit")
builder.add_edge("reconstruct_workflow", "consistency_audit")
builder.add_edge("reconstruct_workflow", "reproducibility_audit")

for audit_node in [
    "novelty_audit",
    "evidence_audit",
    "methodology_audit",
    "consistency_audit",
    "reproducibility_audit",
]:
    builder.add_edge(audit_node, "synthesize_readiness")
```

The synthesis node should execute only after all required audit outputs are available. If an audit fails, it should contribute a blocking or warning finding rather than silently disappear.

### 10.3 Conditional research-completion loop

The assessment graph should route to a completion plan when additional calculations, experiments, controls, documentation, or literature work are needed. The researcher may approve selected tasks. Approved jobs execute outside the graph in the bounded scientific job runner. When results are committed as derived artifacts, a new assessment thread or a new assessment run should re-enter discovery and audit.

```text
readiness synthesis
  → needs_completion
  → human approves selected tasks
  → scientific job runner executes
  → outputs registered as derived artifacts
  → updated knowledge map
  → repeat affected audits
  → revised readiness report
```

Do not create a loop that blindly repeats all audits after every new artifact. The completion plan should identify affected evidence and audit types, enabling targeted re-analysis.

## 11. Human-in-the-loop implementation

Use dynamic interrupts for decisions that require the researcher. The interrupt payload must be JSON-serializable and should contain the question, affected artifact IDs, findings, recommended choices, and required rationale.

```python
from langgraph.types import interrupt


def human_assessment_review(state: AssessmentState):
    report = load_readiness_report(state["readiness_report_id"])

    decision = interrupt({
        "kind": "ASSESSMENT_REVIEW",
        "question": "Review the publication-readiness assessment.",
        "readiness_status": state["readiness_status"],
        "report_id": state["readiness_report_id"],
        "blocking_finding_ids": state.get("blocking_finding_ids", []),
        "choices": [
            "APPROVE_MANUSCRIPT_PLANNING",
            "APPROVE_COMPLETION_PLAN",
            "REQUEST_REASSESSMENT",
            "BLOCK_RUN",
        ],
    })

    decision_id = record_human_decision(
        project_id=state["context"]["project_id"],
        report_id=state["readiness_report_id"],
        decision=decision,
    )

    return {
        "assessment_approval_id": decision_id,
        "human_decision": decision,
    }
```

The API must use the same `thread_id` to resume the graph. The application should never treat an interrupt payload as an approval by itself. Approval must be validated against the allowed choices, user identity, project permissions, and current artifact versions.

## 12. Idempotency and side effects

LangGraph nodes may be re-executed after retries, resumption, or recovery. Therefore, every side-effecting operation must be idempotent or use a durable operation key.

| Operation | Idempotency key |
|---|---|
| Source scan | project ID + source path + scan configuration + file manifest hash |
| Artifact commit | content hash + artifact type + parent IDs |
| Literature metadata fetch | provider + query/identifier + provider response version |
| Scientific job | job specification hash + input artifact IDs + environment lock hash |
| Manuscript block generation | block ID + input claim IDs + prompt/model/code hashes |
| Report generation | report type + input artifact IDs + rule/profile versions |
| Release package | release specification hash + approved artifact IDs |

Never place a non-idempotent file write, job submission, or external mutation immediately before an interrupt unless the operation has an idempotency key and a recorded completion status.

## 13. Node catalog

### Intake and discovery nodes

| Node | Responsibility | Side effects | Output |
|---|---|---|---|
| `validate_boundary` | Validate path, exclusions, permissions, symlinks, and execution authorization | Database record only | intake status |
| `create_manifest` | Scan files and calculate checksums | Source registry writes | source manifest ID |
| `discover` | Parse files, notebooks, code, tables, images, logs, and documentation | Derived artifacts | discovery report ID |
| `classify_domain` | Infer domain and confidence from evidence | Classification record | domain profile ID |
| `build_knowledge_map` | Connect methods, inputs, runs, outputs, figures, and documents | Knowledge-map artifact | map ID |
| `reconstruct_workflow` | Describe the actual project execution path and unresolved links | Workflow artifact | workflow map ID |

### Audit nodes

| Node | Responsibility | Output |
|---|---|---|
| `novelty_audit` | Compare candidate contribution with verified literature and identify gaps | Novelty audit ID |
| `evidence_audit` | Evaluate claims, controls, baselines, uncertainty, and support | Evidence audit ID |
| `methodology_audit` | Check domain-specific methods and reporting completeness | Methodology audit ID |
| `consistency_audit` | Cross-check code, numbers, plots, tables, units, and text | Consistency audit ID |
| `reproducibility_audit` | Check environment, parameters, seeds, inputs, and run records | Reproducibility audit ID |
| `synthesize_readiness` | Apply policy rules and create readiness report | Readiness report ID |

### Manuscript nodes

| Node | Responsibility | Output |
|---|---|---|
| `select_journal_profile` | Load a versioned journal profile | Profile ID/version |
| `build_manuscript_plan` | Create sections, claim allocation, evidence requirements, and word limits | Plan ID |
| `register_claims` | Promote approved candidate claims into manuscript scope | Claim IDs/status |
| `generate_section_blocks` | Generate evidence-linked blocks by section | Block IDs |
| `generate_figures_tables` | Validate or generate captions and tables | Figure/table IDs |
| `citation_audit` | Verify references, DOI metadata, relevance, and citation coverage | Citation audit ID |
| `scientific_qa` | Run factual, numerical, equation, consistency, and reproducibility QA | QA report ID |
| `human_draft_review` | Pause for scientific review and revisions | Decision ID |
| `publication_qa` | Run language, proof, similarity, and journal checks | Publication QA ID |
| `package_release` | Assemble only approved artifacts | Release candidate ID |

## 14. Agent design rules

Agents should be small, typed, and replaceable. An agent should not directly access arbitrary state or the filesystem. It should receive an application service interface and a validated task input.

A base protocol should look like this:

```python
from dataclasses import dataclass
from typing import Generic, TypeVar

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")

@dataclass(frozen=True)
class AgentRequest(Generic[InputT]):
    task_id: str
    project_id: str
    input: InputT
    input_artifact_ids: list[str]
    policy_context: dict
    provenance_context: dict

@dataclass(frozen=True)
class AgentResponse(Generic[OutputT]):
    task_id: str
    status: str
    output: OutputT | None
    output_artifact_ids: list[str]
    evidence_ids: list[str]
    finding_ids: list[str]
    confidence: float | None
    warnings: list[str]
    requires_human_review: bool
    provenance: dict
```

A writer may propose a paragraph, but it may not approve its own paragraph. A fact checker may produce a finding, but it may not erase a conflicting user-provided result. A citation agent may propose a verified reference, but it must not invent missing metadata.

## 15. Model gateway

All LLM and embedding calls should pass through a model gateway. The gateway should provide capability routing rather than allowing each node to construct its own model client.

| Capability | Preferred model class |
|---|---|
| File classification and routing | Small local model or deterministic classifier |
| Literature extraction | Medium model with structured output |
| Scientific interpretation | Strong local model plus deterministic validators |
| Manuscript generation | Strong model constrained by approved claims and evidence |
| Proofreading | Small/medium model plus deterministic style checks |
| Embeddings | Local embedding model |
| Equation assistance | Specialized mathematical model or symbolic tools |

Every model call must record provider/runtime, model identifier, configuration, prompt-template version, input artifact IDs, output hash, and whether the output was accepted or rejected. Model output is always a proposal until validated and approved.

## 16. Persistence design

Use two persistence layers for LangGraph and a separate domain persistence layer.

### 16.1 LangGraph checkpointer

The checkpointer stores thread-scoped graph state so an assessment or manuscript run can resume after interruption or failure. Development may use an in-memory saver, but production-like local use should use SQLite or PostgreSQL-backed persistence because in-memory state is lost on process restart [2].

The `thread_id` should be stable, short, and unrelated to sensitive path strings. Use a generated UUID or a deterministic project/run identifier under the database’s length constraints.

### 16.2 Application store

The store or application database should contain project facts, claims, evidence, artifacts, findings, decisions, journal profiles, and release manifests. The graph should hold only IDs and compact coordination state.

### 16.3 Artifact store

Use a content-addressed filesystem initially. Each artifact should have a manifest with its hash, media type, size, parent IDs, creator, code revision, environment, and creation time. Large scientific outputs should not be serialized into graph state.

## 17. Observability and evaluation

Every graph run should emit structured events:

```json
{
  "event_id": "EVT-0001",
  "run_id": "RUN-0001",
  "thread_id": "THREAD-0001",
  "node": "evidence_audit",
  "event_type": "NODE_COMPLETED",
  "input_artifact_ids": [],
  "output_artifact_ids": [],
  "status": "SUCCEEDED",
  "duration_ms": 0,
  "model_id": null,
  "code_revision": "git-revision",
  "created_at": "timestamp"
}
```

Use LangGraph event streaming for UI progress and graph-level debugging. Optional LangSmith tracing can be added for development and evaluation, but the local application must continue to function without sending research data externally unless the user explicitly enables it [1].

Evaluation should use golden projects and adversarial fixtures. The evaluation dataset should contain known answers for file classification, claim support, numerical consistency, readiness blockers, and journal checks.

## 18. Safety and scientific-integrity policy

The policy engine must block or require human review for the following conditions:

| Condition | Default action |
|---|---|
| Raw source mutation detected | Block dependent workflow |
| Fabricated or unverifiable DOI | Block automatic citation insertion |
| Central claim without evidence | Block readiness |
| Figure contradicts central numerical claim | Block release |
| Missing critical domain parameters | Warning or block according to profile |
| Model proposes new number not present in evidence | Block until human verifies |
| Interpretation presented as established fact | Require revision |
| Similarity detected | Report overlap; do not label plagiarism automatically |
| Failed scientific job | Record failure; never invent substitute output |
| Unclear source provenance | Mark unverified and request review |
| Submission action requested | Generate package only; require manual action |

## 19. Project assessment graph: detailed flow

```text
START
  → intake_request
  → validate_boundary
      ├── invalid_path → BLOCKED
      ├── execution_not_authorized → discovery_without_execution
      └── valid → create_manifest
  → discover_artifacts
  → classify_domain
      ├── high_confidence → selected_profile
      ├── low_confidence → HUMAN_DOMAIN_CONFIRMATION
      └── unsupported → generic_profile + warning
  → build_knowledge_map
  → reconstruct_workflow
  → parallel audits
      ├── novelty
      ├── evidence
      ├── methodology
      ├── consistency
      └── reproducibility
  → synthesize_readiness
  → HUMAN_ASSESSMENT_REVIEW
      ├── APPROVE_MANUSCRIPT_PLANNING → manuscript handoff
      ├── APPROVE_COMPLETION_PLAN → completion plan
      ├── REQUEST_REASSESSMENT → targeted audit loop
      └── BLOCK_RUN → terminal blocked state
END
```

The discovery graph should complete even if the researcher has not authorized code execution. In that case it should clearly state that the assessment is based on static evidence and existing outputs, not newly reproduced calculations.

## 20. Manuscript production graph: detailed flow

```text
approved assessment
  → select journal profile
  → build manuscript plan
  → human approves outline
  → register approved claims
  → generate Introduction / Methods / Results / Discussion / Conclusion
  → generate Abstract after main text stabilization
  → validate figures, tables, equations, and citations
  → run scientific QA
  → human draft review
      ├── revise → targeted revision loop
      └── continue → language/proof QA
  → journal compliance
  → similarity classification
  → final readiness report
  → human release approval
  → package release
```

Section writers must consume approved claims and evidence references. The system should prefer controlled placeholders or `MISSING_EVIDENCE` findings over fluent invention. The Abstract graph should run late, after the core manuscript has stabilized, so that it summarizes approved content rather than anticipating results.

## 21. Journal profile schema

```yaml
journal_profile:
  profile_id: example-journal
  version: "2026-01"
  source_url: ""
  effective_date: ""
  publisher: ""
  journal_name: ""
  article_types:
    - research_article
  manuscript:
    required_sections: []
    word_limit: null
    abstract_limit: null
    keywords_required: true
  citations:
    style: ""
    doi_required: true
    reference_order: ""
  figures:
    formats: []
    resolution: null
    graphical_abstract: false
  statements:
    data_availability: required
    code_availability: recommended
    conflicts_of_interest: required
    author_contributions: required
  package:
    cover_letter: required
    supplementary_information: optional
  rules:
    - rule_id: abstract_length
      severity: blocking
      validator: word_count
```

Profiles must be versioned and should retain their source and effective date. A missing or stale rule should produce `NOT_VERIFIABLE`, not a false pass.

## 22. Research-completion plan schema

```yaml
completion_plan:
  plan_id: PLAN-0001
  project_id: PROJECT-0001
  tasks:
    - task_id: COMP-0001
      title: "Run convergence test"
      reason: "Central energy comparison lacks convergence evidence"
      category: ANALYSIS
      priority: REQUIRED
      estimated_effort: MEDIUM
      required_inputs: [ART-0001, ART-0002]
      expected_outputs: ["convergence table", "convergence plot"]
      affected_claim_ids: [CLM-0001]
      requires_human_approval: true
      execution_status: PROPOSED
```

The system must not invent effort estimates as precise time commitments. Use qualitative categories unless the researcher supplies a benchmarked execution model.

## 23. Testing plan

### Unit tests

Test parsers, checksum logic, reducers, routing functions, policy rules, schema validation, journal profile validators, and artifact lineage independently.

### Graph tests

Invoke each graph with controlled state and assert node routing, terminal states, interrupt payloads, resume behavior, and accumulated IDs. Test both successful and failed branches.

### Persistence tests

Restart the process between graph steps and verify that the assessment or manuscript run resumes from the persistent checkpointer. Test retention and cleanup policies without deleting active runs.

### Safety tests

Confirm that discovery cannot write to the source folder, that symlinks outside the allowed root are handled correctly, that arbitrary commands are not executed without authorization, and that output files cannot replace raw artifacts.

### Scientific integrity fixtures

Include projects containing fabricated DOI records, contradictory numbers, inconsistent units, missing DFT parameters, leakage-prone ML splits, absent quantum shot counts, orphaned figures, unsupported interpretations, and incomplete provenance.

### Golden end-to-end project

Maintain one complete demonstration project for which the researcher knows the expected project map, major claims, evidence links, readiness blockers, manuscript outline, and final QA findings. Any change to prompts, model versions, parsers, or policies should be evaluated against this project.

## 24. Implementation phases

### Phase 0 — LangGraph skeleton and local persistence

Create the Python package, configuration system, domain models, graph state schemas, SQLite application database, local artifact store, and a minimal compiled graph. Add one interrupt/resume test and one source-manifest test.

**Exit condition:** A project run can be created, checkpointed, interrupted, resumed, and completed without an LLM.

### Phase 1 — Read-only discovery graph

Implement path validation, source scanning, checksums, file classification, basic text/table/image extraction, notebook inspection, environment detection, and discovery reports.

**Exit condition:** The original project directory remains unchanged and the discovery report is reproducible.

### Phase 2 — Knowledge map and domain profiles

Implement knowledge-map records, relationship extraction, domain classification, DFT/materials checks, AI/ML checks, and quantum checks. Add human confirmation when domain confidence is low.

**Exit condition:** A researcher can inspect and correct the project map, and the selected domain profile is recorded.

### Phase 3 — Scientific audit and readiness graph

Implement parallel audits, claim/evidence records, findings, readiness synthesis, approval interrupts, and research-completion plans.

**Exit condition:** The system produces a defensible readiness report and never labels unsupported claims as established results.

### Phase 4 — Manuscript planning and controlled writing

Implement journal profiles, outline generation, claim allocation, section writers, figure/table registration, citation validation, and block-level provenance.

**Exit condition:** A draft manuscript can be generated only from approved assessment artifacts and claims.

### Phase 5 — Scientific and publication QA

Implement fact checking, equation and unit checks, consistency, reproducibility, peer-review simulation, language and proof checks, similarity classification, and journal compliance.

**Exit condition:** Critical findings block release and every release candidate includes a complete readiness report.

### Phase 6 — Release and review-cycle automation

Implement response-to-reviewers, revision tracking, cover letter, supplement, graphical abstract, release manifests, and package generation. Keep actual submission manual.

**Exit condition:** A release package can be reconstructed and audited from stored artifact IDs and approvals.

## 25. MVP definition

The MVP should not implement every agent. It should implement the following vertical slice:

```text
project path
  → read-only discovery
  → manifest and checksums
  → knowledge map
  → one selected domain profile
  → evidence and methodology audit
  → readiness report
  → human approval
  → manuscript outline
```

The MVP should not include autonomous publication, unrestricted code execution, automatic external submissions, a distributed microservice architecture, or a promise of publication probability. Once the MVP is trusted on one real project, add the domain profiles and writing agents incrementally.

## 26. Recommended first command-line interface

```bash
# Register a project without running code
msys project intake \
  --path /absolute/path/to/project \
  --domain auto \
  --mode assessment \
  --read-only \
  --target-journal optional-journal-id

# Inspect current run state
msys run status --run-id RUN-0001

# Display pending human approvals
msys approval list --project-id PROJECT-0001

# Resume an approved interrupt
msys approval respond \
  --approval-id APR-0001 \
  --decision APPROVE_COMPLETION_PLAN

# Generate the readiness report
msys report readiness --run-id RUN-0001 --output reports/readiness.md

# Start manuscript planning only after assessment approval
msys manuscript plan \
  --assessment-id ASSESS-0001 \
  --journal example-journal
```

The CLI should expose safe defaults. A command that runs code or launches expensive scientific calculations should require an explicit flag and a separate approval path.

## 27. Final architectural recommendation

Use LangGraph. It is a strong fit for the graph-shaped, stateful, interruptible workflow required by this system. Do not use LangGraph alone. Pair it with a structured application database, content-addressed artifact store, append-only provenance records, policy engine, bounded scientific job runner, model gateway, and optional observability service.

The correct implementation order is:

```text
LangGraph skeleton
  → safe intake
  → deterministic discovery
  → knowledge map
  → domain audits
  → claim–evidence readiness
  → human approval
  → manuscript planning
  → controlled writing
  → QA and release
```

This architecture is realistic because it avoids the most common failure: building many LLM agents before establishing reliable project discovery, evidence storage, provenance, and approval boundaries. LangGraph should coordinate the system, but scientific truth must remain grounded in the project artifacts, verified literature, deterministic analysis, and the researcher’s explicit decisions.

## References

[1]: https://docs.langchain.com/oss/python/langgraph/overview "LangGraph overview — official documentation"

[2]: https://docs.langchain.com/oss/python/langgraph/persistence "LangGraph persistence — official documentation"

[3]: https://docs.langchain.com/oss/python/langgraph/interrupts "LangGraph interrupts — official documentation"

[4]: https://docs.langchain.com/oss/python/langgraph/graph-api "LangGraph Graph API — official documentation"
