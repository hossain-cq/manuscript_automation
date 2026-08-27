# Automated Peer-Review Simulation and Revision Loop

## LangGraph implementation specification

## 1. Purpose and boundaries

The peer-review subsystem should act as a **pre-submission quality-control process**, not as a prediction of a real journal’s decision. It should expose weaknesses, missing evidence, overclaiming, unclear methods, statistical problems, reproducibility gaps, citation issues, and presentation problems before human submission.

The system should simulate multiple reviewer perspectives, but it should not pretend that several LLM personas are independent human reviewers. Their outputs are model-generated critiques and must be traced, triaged, validated, and reviewed by the researcher.

## 2. Required workflow

```text
approved manuscript blocks
  → reviewer-context construction
  → independent structured reviewer simulations
  → comment normalization and deduplication
  → validity triage
  → human revision-plan approval
  → section/block-level revision proposals
  → deterministic revision verification
  → human revision approval
  → response-to-reviewers generation
  → human release approval
  → manuscript revision round or release candidate
```

The system should revise blocks, not overwrite one monolithic manuscript string. Each block must retain its prior version, source evidence, claims, literature IDs, authoring model, prompt hash, and revision lineage.

## 3. Reviewer personas

| Persona | Primary review focus |
|---|---|
| Scientific expert | Scientific validity, contribution, mechanism, domain assumptions, interpretation |
| Critical reviewer | Unsupported claims, missing controls, weak comparisons, alternative explanations |
| Statistical/computational reviewer | Uncertainty, convergence, data splits, baselines, leakage, reproducibility, numerical validity |
| Journal reviewer | Scope, article type, structure, significance, clarity, journal-specific requirements |
| Hostile reviewer | Serious failure modes and the strongest plausible objections, without inventing unrelated demands |

The system should run the same manuscript through each persona using the same evidence packet but different review instructions. Persona outputs should be stored separately to make disagreement visible.

## 4. Review comment contract

Every substantive review comment should point to an affected block, section, claim, or evidence record.

```python
class ReviewComment(BaseModel):
    comment_id: str
    reviewer_id: str
    severity: Literal["MAJOR", "MINOR", "TECHNICAL", "EDITORIAL"]
    category: Literal[
        "SCIENTIFIC_VALIDITY", "METHODS", "EVIDENCE", "STATISTICS",
        "REPRODUCIBILITY", "LITERATURE", "OVERCLAIM", "FIGURE_TABLE",
        "JOURNAL_FIT", "LANGUAGE", "OTHER"
    ]
    text: str
    affected_section_ids: list[str]
    affected_block_ids: list[str]
    affected_claim_ids: list[str]
    evidence_ids: list[str]
    validity: Literal["VALID", "PARTIALLY_VALID", "UNCLEAR", "INVALID"]
    status: Literal[
        "OPEN", "UNDER_REVIEW", "ADDRESSED", "PARTIALLY_ADDRESSED",
        "REJECTED_WITH_JUSTIFICATION"
    ]
```

A review comment without an affected location should be treated as low-confidence and routed to human triage. The review agent should not create a technical requirement merely because it sounds plausible.

## 5. Comment triage

Triage is a separate step from review generation. It should classify each comment as valid, partially valid, unclear, or invalid based on the supplied manuscript and evidence.

| Triage result | Action |
|---|---|
| Valid | Create a revision task or research-completion task |
| Partially valid | Narrow the task and preserve the scientifically justified part |
| Unclear | Human review before revision |
| Invalid | Record a reasoned rejection; do not silently ignore it |

Major scientific comments should not be “fixed” by wording changes if the underlying evidence is missing. They should become research-completion tasks, such as adding a control, rerunning convergence, adding an external test, or revising the claim scope.

## 6. Revision plan approval

Before changing the manuscript, the researcher should approve the revision plan. The approval view should show:

| Field | Purpose |
|---|---|
| Comment ID | Stable reviewer-comment identity |
| Affected block/claim | Exact target of revision |
| Proposed action | Wording change, evidence addition, figure change, analysis task, or rejection |
| Scientific consequence | Whether the central claim changes |
| Evidence required | Existing evidence IDs or new work required |
| Risk | Possibility of overclaiming or changing interpretation |

A revision plan should support explicit researcher edits. The system should not assume that all comments should be accepted.

## 7. Controlled revision rules

The revision agent should receive only the affected block, comment, approved claims, approved evidence, literature records, and journal profile. It should not receive authority to modify unrelated sections.

The revision agent must follow these rules:

1. It may clarify or narrow an existing statement.
2. It may reorganize text while preserving claim and evidence links.
3. It may add a limitation supported by the review and existing evidence.
4. It may not add a new numerical value without an approved evidence record.
5. It may not add a new experiment, calculation, dataset, method, or citation as if it already exists.
6. If the reviewer requests new scientific work, it must abstain and create a research-completion task.
7. It may not delete a central claim without recording the change and requiring human approval.
8. It must return a change summary and a response-to-reviewers draft.

## 8. Deterministic revision verification

Revision verification must run after every revision proposal.

| Check | Purpose |
|---|---|
| Claim-ID validity | Prevent unapproved claims from entering text |
| Evidence-ID validity | Ensure every material claim remains grounded |
| Literature-ID validity | Prevent fabricated references |
| Numeric-token diff | Detect newly introduced numbers requiring verification |
| Scope diff | Detect new universal or causal language |
| Block lineage | Preserve original block and revision ancestry |
| Section boundary | Ensure the revision affects only approved sections |
| Word/format limits | Enforce journal profile constraints |
| Contradiction scan | Detect conflicts with other approved blocks |
| Citation placement | Verify citations correspond to literature records |

Any new numeric value, new citation, unsupported claim, or central-scope change should trigger human review. A deterministic checker cannot prove that prose is scientifically correct, but it can prevent many silent violations.

## 9. Revision-loop stopping rules

The loop should not run indefinitely. Use explicit stopping conditions:

```text
stop_successfully when:
  no unresolved MAJOR comments remain
  and no failed revision verification exists
  and human approves the response-to-reviewers document

stop_with_research_tasks when:
  remaining comments require new experiments, calculations, data, or external validation

stop_blocked when:
  critical integrity issue remains
  or a central claim cannot be supported
  or the revision introduces unverified content

stop_after_limit when:
  review_round >= configured_max_rounds
```

A new review round should be triggered only when the researcher approves it. The system should not automatically keep rewriting after a reviewer comment remains unresolved.

## 10. Response-to-reviewers document

The system should produce a structured response with one entry per comment.

```yaml
response_entry:
  comment_id: REV-0001
  reviewer_id: reviewer-statistical-computational
  decision: ADDRESSED | PARTIALLY_ADDRESSED | REJECTED_WITH_JUSTIFICATION | RESEARCH_REQUIRED
  response: "Respectful response grounded in the approved evidence."
  manuscript_changes:
    - block_id: BLOCK-0004
      description: "Narrowed the generalization statement."
  new_evidence_ids: []
  unresolved_reason: null
```

The response must never claim that a new analysis was performed unless a new approved artifact and provenance record exist. If a request is rejected, the response should explain why using the project scope and evidence, not dismiss the reviewer.

## 11. Review-round data model

Each review round should record the manuscript revision ID, journal profile version, reviewer model IDs, prompt versions, reviewer reports, comment statuses, revision plan, revision proposals, verification findings, response-to-reviewers version, and human approvals.

This creates a reproducible audit trail:

```text
manuscript_release
  → review_round
  → reviewer_report
  → review_comment
  → revision_plan
  → revision_proposal
  → verification
  → approved_block_version
  → response_to_reviewers
```

## 12. LangGraph graph structure

### Drafting graph

```text
START
  → write_next_section
  → write_next_section until queue exhausted
  → human_draft_review
  → END or BLOCKED
```

The graph uses a section queue and a reducer-backed block collection. Each section writer is constrained by its section specification and approved claim/evidence IDs.

### Review graph

```text
START
  → reviewer[scientific expert]
  → reviewer[critical]
  → reviewer[statistics/computational]
  → reviewer[journal]
  → reviewer[hostile]
  → triage
  → human revision-plan approval
  → revise affected blocks
  → verify revisions
  → human revision approval
  → response-to-reviewers
  → human response approval
  → END or another approved review round
```

The supplied Python implementation uses sequential reviewer calls for simplicity and deterministic traceability. It can later be converted to a LangGraph fan-out/fan-in structure with reducer-backed report and comment lists.

## 13. Model selection

Use a lower-cost model for ordinary section drafting and comment normalization, and a stronger reasoning model for critical-review simulation, statistical review, and difficult revision decisions. Model IDs should be configurable and recorded per artifact. The model catalog should be checked at deployment time rather than hard-coded permanently.

A practical starting configuration is:

| Task | Starting model |
|---|---|
| Section drafting | `gpt-5-mini` |
| Reviewer simulation | `gpt-5` or a stronger configured reasoning model |
| Revision wording | `gpt-5-mini` |
| High-risk disagreement resolution | Stronger model plus human review |

Model output is not scientific evidence. Evidence must continue to come from the project artifact and literature registries.

## 14. Quality evaluation

Evaluate the subsystem with expert-authored manuscript cases containing known weaknesses.

| Evaluation | Metric |
|---|---|
| Evidence preservation | Percentage of approved claims retaining valid evidence links |
| Hallucination prevention | Unsupported numeric/citation insertion rate |
| Comment localization | Percentage of comments linked to correct blocks/claims |
| Triage quality | Expert agreement on valid/invalid/unclear classification |
| Revision usefulness | Expert rating of whether revisions address comments without overclaiming |
| Regression safety | Percentage of unrelated blocks unchanged |
| Release correctness | Percentage of releases passing format, provenance, and approval gates |
| Abstention quality | Correct refusal when new work is required |

The most important production metrics are unsupported-content prevention and traceability, not prose fluency.

## 15. Human approvals

Human approval is required before:

- The first draft is sent to simulated reviewers.
- A revision plan changes a central claim.
- A revision adds or removes evidence or references.
- A new number, method, experiment, or conclusion appears.
- A major reviewer comment is rejected.
- The response-to-reviewers document is finalized.
- A manuscript release package is generated.

## 16. Final recommendation

The correct design is a **versioned, evidence-linked drafting and review system**. Let LangGraph manage the workflow, interrupts, retries, and revision routing. Let structured LLMs generate proposals and critiques. Let deterministic validators enforce identifiers, numeric changes, journal constraints, and block lineage. Let a human approve scientific decisions and release transitions.

The system should not claim that an automated peer review replaces peer review. Its honest output is:

> “The simulated reviewers identified three major evidence issues, two valid presentation issues, and one comment requiring new calculations. The wording revisions passed identifier and provenance checks, but the manuscript is not release-ready until the calculation task is completed and the researcher approves the response-to-reviewers package.”
