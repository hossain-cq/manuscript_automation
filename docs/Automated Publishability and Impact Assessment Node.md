# Automated Publishability and Impact Assessment Node

## Design for LangGraph-based scientific manuscript system

## 1. Core principle

The node should not attempt to predict whether a journal will accept a manuscript. That would create false precision and would confuse scientific readiness with editorial decision-making. Instead, it should produce three separate outputs:

1. **Evidence Readiness:** whether the available project evidence is sufficient to support a defensible manuscript draft and its central claims.
2. **Publication Readiness:** whether the project has passed the required scientific, methodological, reproducibility, citation, and presentation gates for the selected manuscript stage.
3. **Potential Significance:** how important, novel, useful, generalizable, or field-relevant the candidate contribution appears based on verified literature and evidence.

The system may display numerical scores, but each score must remain a **diagnostic indicator** with an evidence trail. The final status should be rule-based and explainable, not a simple average.

> **Do not implement:** `publishability = LLM_score(project)`.  
> **Implement:** `readiness = deterministic evidence gates + structured LLM assessment + calibrated uncertainty + human approval`.

## 2. Recommended assessment architecture

The assessment should be implemented as a LangGraph subgraph with deterministic feature extraction, independent LLM assessments, aggregation, blocking-policy evaluation, report generation, and a human approval interrupt.

```text
Project Knowledge Map
       ↓
Claim and evidence registry
       ↓
Deterministic feature extraction
       ├── evidence coverage
       ├── reproducibility completeness
       ├── numerical/statistical robustness
       ├── domain checks
       ├── cross-artifact consistency
       ├── literature positioning
       └── journal/profile completeness
       ↓
LLM structured assessments
       ├── contribution assessor
       ├── scientific-evidence assessor
       ├── critical reviewer
       └── journal-fit assessor
       ↓
Calibration and disagreement analysis
       ↓
Blocking-policy evaluation
       ↓
Readiness and impact report
       ↓
Human approval interrupt
```

The LLM should not directly inspect an arbitrary folder and return a score. It should receive a compact evidence packet containing artifact IDs, extracted facts, relevant source passages, numerical summaries, audit findings, and explicit uncertainty.

## 3. Separate dimensions

The system should report a vector of dimension scores rather than one opaque number.

| Dimension | Primary question | Main evidence source |
|---|---|---|
| Scientific contribution | Is there a clearly stated contribution candidate? | Approved claim registry and literature comparison |
| Evidence sufficiency | Does the evidence support the central claims? | Claim–evidence graph and domain audits |
| Methodological rigor | Are methods sufficiently specified and valid? | Methodology audit and project knowledge map |
| Validation strength | Are baselines, controls, convergence, uncertainty, or comparison studies adequate? | Deterministic domain metrics |
| Reproducibility | Could another researcher reconstruct the work? | Environment, code, parameters, inputs, and run records |
| Internal consistency | Do code, outputs, plots, tables, units, and text agree? | Cross-artifact consistency checks |
| Literature positioning | Is the difference from prior work sufficiently established? | Verified literature records and comparison matrix |
| Presentation readiness | Are figures, tables, captions, equations, and terminology usable? | Figure/table and manuscript-preparation checks |
| Journal fit | Does the candidate contribution match the selected article type and scope? | Versioned journal profile plus LLM explanation |
| Potential significance | Could the contribution matter to the field or application? | Literature context, contribution type, utility, and generality |

The first nine dimensions are primarily readiness dimensions. **Potential significance** should be reported separately from readiness because an important idea can be underdeveloped, and a technically complete project can still have limited impact.

## 4. Hard blockers before scoring

The node must evaluate hard blockers first. A high score must not compensate for a critical integrity failure.

| Blocker | Default result |
|---|---|
| Central claim has no linked evidence | `INSUFFICIENT_EVIDENCE` |
| Raw data or source artifact was modified unexpectedly | `BLOCKED` |
| Fabricated or unverifiable critical reference | `BLOCKED` for affected claim |
| Figure or table contradicts a central numerical claim | `NEEDS_REVIEW` or `BLOCKED` |
| Critical domain parameters are missing | `NEEDS_ADDITIONAL_ANALYSIS` or `INSUFFICIENT_EVIDENCE` |
| Statistical or validation procedure invalid for the stated claim | `NEEDS_ADDITIONAL_ANALYSIS` |
| Project workflow cannot be reconstructed | `INSUFFICIENT_EVIDENCE` |
| LLM assessment relies on evidence not present in the evidence packet | Reject assessment and re-run |
| Model invents numerical values, experiments, or references | Reject output and create integrity finding |
| Final conclusion lacks human approval | Cannot reach release-ready status |

Hard blockers are represented as structured `Finding` records. They should be evaluated by deterministic policy code, not by an LLM.

## 5. Deterministic empirical metrics

The empirical layer converts project artifacts and audit results into measurable features. A feature must include its value, calculation method, input artifact IDs, validity status, and missing-data reason where applicable.

```python
from pydantic import BaseModel, Field
from typing import Literal

class EmpiricalFeature(BaseModel):
    feature_id: str
    value: float | None
    scale_min: float = 0.0
    scale_max: float = 1.0
    status: Literal["VALID", "MISSING", "NOT_APPLICABLE", "INVALID"]
    method: str
    input_artifact_ids: list[str] = Field(default_factory=list)
    supporting_finding_ids: list[str] = Field(default_factory=list)
    uncertainty: float | None = None
    notes: str | None = None
```

### 5.1 Evidence coverage metrics

For every claim, calculate whether there is direct evidence, indirect evidence, literature evidence, or no evidence. Weight central claims more heavily than contextual claims.

```text
central_claim_coverage =
    weighted_supported_central_claims / weighted_total_central_claims

major_claim_gap_rate =
    unsupported_or_weak_major_claims / total_major_claims

provenance_completeness =
    claims_with_complete_source_lineage / total_claims
```

A claim should not count as fully supported merely because it has a citation. The evidence relationship must match the claim type. A literature citation may support a background statement but not the researcher’s new numerical result.

### 5.2 Reproducibility metrics

The reproducibility feature should be a checklist-based coverage measure, not a subjective LLM score.

```text
reproducibility_completeness =
    weighted_present_required_items / weighted_required_items
```

Required items depend on the domain. Examples include software versions, parameter files, random seeds, dataset provenance, model configuration, convergence criteria, hardware, code revision, and exact input artifacts.

### 5.3 Consistency metrics

Calculate the rate of successful cross-checks across the project.

```text
consistency_score =
    passed_consistency_checks / applicable_consistency_checks
```

Examples include values extracted from plots versus tables, values in reports versus source outputs, units across files, figure labels versus source data, and stated model parameters versus executed configuration.

A single central contradiction should still create a blocker even if the aggregate consistency score is high.

### 5.4 Statistical and validation metrics

The system should not assume that one metric is sufficient. It should detect whether the project contains appropriate uncertainty and validation evidence for the claim type.

Possible features include:

| Feature | Example calculation |
|---|---|
| Baseline coverage | Number of required baseline comparisons present |
| Control coverage | Required negative/positive controls present |
| Repeatability | Repeated runs, seeds, or measurements available |
| Uncertainty reporting | Confidence intervals, standard deviations, error bars, or convergence ranges |
| External validation | Performance on independent data or systems |
| Ablation coverage | Key components tested independently |
| Sensitivity coverage | Important parameters or assumptions varied |
| Statistical validity | Domain-specific test and sample-size checks |

These features should be reported as `VALID`, `MISSING`, or `NOT_APPLICABLE`. Missing evidence should not be silently converted to zero without explanation.

## 6. Domain-specific empirical metrics

### 6.1 DFT and computational materials

The DFT profile should evaluate whether the central conclusions are supported by appropriate computational checks.

| Area | Example empirical features |
|---|---|
| Numerical convergence | Cutoff, k-point, cell, and convergence threshold studies present |
| Structural validity | Relaxation convergence, forces, stress, geometry checks |
| Physical stability | Formation energy references, phonons, stability or metastability analysis where relevant |
| Model sensitivity | Functional, pseudopotential, magnetic, spin, or dispersion sensitivity where material to the claim |
| Comparison | Literature or reference-system comparison using compatible definitions and units |
| Reproducibility | Input decks, code version, pseudopotentials, parameters, and environment recorded |
| Result completeness | Required bands, DOS, phonons, formation energies, or other outputs exist for the stated claim |

A convergence plot alone should not be counted as proof that every conclusion is converged. The feature extractor must map the convergence result to the specific claim and observable.

### 6.2 AI/ML

The AI/ML profile should evaluate the reliability of the experimental design and reported performance.

| Area | Example empirical features |
|---|---|
| Data provenance | Dataset source, licensing, preprocessing, and version recorded |
| Leakage control | Train/validation/test separation and duplicate detection |
| Baselines | Relevant classical, published, or naive baselines present |
| Repeated runs | Multiple seeds, folds, or repetitions where needed |
| Uncertainty | Confidence intervals, variance, calibration, or uncertainty estimates |
| Generalization | External validation, temporal split, scaffold split, or out-of-distribution test where relevant |
| Error analysis | Per-class, subgroup, worst-case, or failure-mode analysis |
| Ablations | Major features, components, or design choices tested |
| Reporting completeness | Hyperparameters, optimizer, scheduler, hardware, and code version recorded |

A high test-set metric must not dominate the score if leakage, weak baselines, or missing uncertainty is detected.

### 6.3 Quantum computing and quantum chemistry

The quantum profile should distinguish mathematical correctness, simulation evidence, resource feasibility, and hardware evidence.

| Area | Example empirical features |
|---|---|
| Problem specification | Hamiltonian, mapping, qubit count, active space, and basis information |
| Algorithm specification | Ansatz, optimizer, initialization, stopping criteria, and iteration history |
| Statistical evidence | Shots, repeated measurements, confidence intervals, and variance |
| Noise treatment | Noise model, mitigation method, backend, and calibration context |
| Baselines | Exact diagonalization, classical solver, tensor-network, or other relevant baseline |
| Scaling | Qubit, circuit-depth, shot, runtime, or memory scaling |
| Reproducibility | Seed, simulator/backend, software versions, and complete configuration |
| Hardware distinction | Clear separation between simulated, emulated, and physical hardware results |

A simulator result should never be described as a hardware demonstration unless hardware evidence is separately verified.

## 7. LLM assessment roles

The LLM layer should perform bounded interpretation, not replace deterministic checks. Use separate structured assessments to reduce self-confirming reasoning.

| Assessor | Question |
|---|---|
| Contribution assessor | What contribution candidates are supported by the evidence? |
| Evidence assessor | Which claims are strongly, weakly, or not supported? |
| Critical reviewer | What would a skeptical expert challenge? |
| Literature assessor | How well is the work differentiated from verified prior work? |
| Journal-fit assessor | Which article types and journal profiles fit the contribution? |
| Impact assessor | What is the potential scientific or practical significance, and what assumptions limit it? |

These are not independent truths. They are structured opinions that must cite evidence IDs and identify uncertainty. The aggregator should detect disagreements rather than hiding them.

## 8. Structured LLM output schema

The LLM must return JSON validated by Pydantic or JSON Schema. It should not return only a score.

```python
from pydantic import BaseModel, Field
from typing import Literal

AssessmentLabel = Literal[
    "STRONGLY_SUPPORTED",
    "SUPPORTED_WITH_LIMITATIONS",
    "WEAKLY_SUPPORTED",
    "UNSUPPORTED",
    "NOT_ASSESSABLE",
]

class ClaimAssessment(BaseModel):
    claim_id: str
    label: AssessmentLabel
    score: float = Field(ge=0.0, le=1.0)
    evidence_ids: list[str]
    reasoning: str
    limitations: list[str]
    missing_evidence: list[str]
    contradiction_finding_ids: list[str]
    confidence: float = Field(ge=0.0, le=1.0)

class ContributionCandidate(BaseModel):
    contribution_id: str
    statement: str
    contribution_type: Literal[
        "METHOD",
        "DATASET",
        "MATERIAL_OR_SYSTEM",
        "BENCHMARK",
        "MECHANISTIC_INSIGHT",
        "APPLICATION",
        "NEGATIVE_RESULT",
        "OTHER",
    ]
    novelty_label: Literal[
        "DISTINCT_FROM_VERIFIED_PRIOR_WORK",
        "PARTIALLY_DISTINCT",
        "UNCLEAR",
        "NOT_SUPPORTED",
    ]
    evidence_ids: list[str]
    literature_record_ids: list[str]
    limitations: list[str]
    confidence: float = Field(ge=0.0, le=1.0)

class LLMReadinessAssessment(BaseModel):
    assessor_id: str
    scientific_contribution: float = Field(ge=0.0, le=1.0)
    evidence_sufficiency: float = Field(ge=0.0, le=1.0)
    methodological_rigor: float = Field(ge=0.0, le=1.0)
    validation_strength: float = Field(ge=0.0, le=1.0)
    reproducibility: float = Field(ge=0.0, le=1.0)
    literature_positioning: float = Field(ge=0.0, le=1.0)
    potential_significance: float = Field(ge=0.0, le=1.0)
    claim_assessments: list[ClaimAssessment]
    contribution_candidates: list[ContributionCandidate]
    major_risks: list[str]
    evidence_ids: list[str]
    finding_ids: list[str]
    confidence: float = Field(ge=0.0, le=1.0)
    abstain: bool = False
```

The prompt must instruct the model to abstain when the evidence packet is incomplete, to never invent references or numbers, and to cite only supplied IDs. A response containing an unknown evidence ID should fail validation and be rejected.

## 9. Evidence packet for the LLM

The LLM should receive a bounded, traceable packet such as:

```yaml
assessment_packet:
  project_id: PROJECT-0001
  knowledge_map_id: MAP-0001
  domain_profile_id: ai_ml.v1
  candidate_claims:
    - claim_id: CLM-0001
      text: "..."
      type: RESULT
      importance: CENTRAL
  evidence_items:
    - evidence_id: EVD-0001
      type: COMPUTATIONAL_RESULT
      source_artifact_id: ART-0001
      location: "results.json:path.metrics.rmse"
      extracted_value: 0.123
      units: "eV"
      extraction_confidence: 0.99
  deterministic_features:
    - feature_id: train_test_leakage_check
      value: 0.0
      status: VALID
      method: duplicate_and_overlap_analysis_v1
  audit_findings:
    - finding_id: FIND-0001
      severity: HIGH
      rule_id: missing_external_validation
      message: "..."
  literature_records:
    - literature_id: LIT-0001
      verified: true
      title: "..."
      doi: "..."
      comparison_notes: "..."
```

The LLM should not receive the entire filesystem by default. Retrieval should be claim-focused and source-located. If more evidence is needed, the assessor should return a structured request for specific evidence IDs or parser operations.

## 10. Score aggregation

Use deterministic aggregation after validating LLM outputs. Do not let the LLM choose the final weights.

### 10.1 Normalize empirical features

Each feature should be normalized to `[0, 1]` using an explicit rule. Do not normalize arbitrary scientific quantities without domain meaning.

```python
def coverage(present: int, required: int) -> float | None:
    if required <= 0:
        return None
    return min(1.0, max(0.0, present / required))
```

### 10.2 Combine empirical and LLM evidence

For dimensions with strong deterministic metrics, use empirical metrics as the primary component. For dimensions involving interpretation, use the LLM as a bounded secondary component.

```text
readiness_dimension_score =
    empirical_weight * empirical_score
    + llm_weight * calibrated_llm_score
```

Recommended starting weights:

| Dimension | Empirical weight | LLM weight |
|---|---:|---:|
| Evidence sufficiency | 0.75 | 0.25 |
| Methodological rigor | 0.75 | 0.25 |
| Validation strength | 0.85 | 0.15 |
| Reproducibility | 0.90 | 0.10 |
| Internal consistency | 0.95 | 0.05 |
| Literature positioning | 0.45 | 0.55 |
| Potential significance | 0.25 | 0.75 |
| Journal fit | 0.50 | 0.50 |

These are initial engineering weights, not universal scientific truths. They should be versioned, evaluated on golden projects, and changed only through a documented policy revision.

### 10.3 Apply coverage and uncertainty penalties

A score based on weak evidence should not look as strong as a score based on broad evidence. Use a coverage factor and a disagreement factor.

```text
adjusted_score =
    raw_score
    × evidence_coverage
    × (1 - unresolved_critical_uncertainty_penalty)
```

If independent LLM assessors disagree strongly, do not simply average them. Record the disagreement and lower confidence or require human review.

```text
llm_disagreement = mean pairwise absolute score difference
confidence = base_confidence × (1 - llm_disagreement)
```

### 10.4 Readiness status rules

The final status should be determined by rules rather than a score threshold alone.

```python
def readiness_status(
    *,
    blockers: list[str],
    evidence_score: float,
    reproducibility_score: float,
    validation_score: float,
    central_claim_coverage: float,
    high_risk_uncertainty: bool,
) -> str:
    if blockers:
        return "BLOCKED"
    if central_claim_coverage < 0.70:
        return "INSUFFICIENT_EVIDENCE"
    if evidence_score < 0.60 or validation_score < 0.55:
        return "NEEDS_ADDITIONAL_ANALYSIS"
    if reproducibility_score < 0.50:
        return "DRAFTABLE_WITH_WARNINGS"
    if high_risk_uncertainty:
        return "DRAFTABLE_WITH_WARNINGS"
    return "READY_FOR_MANUSCRIPT"
```

Thresholds must be domain-profile configuration, not hard-coded assumptions. A quantum hardware study, DFT benchmark, and exploratory ML paper should not use identical requirements.

## 11. Potential impact assessment

Impact must be assessed as **potential significance**, not as predicted citations or guaranteed influence. The node should evaluate the following dimensions:

| Impact dimension | Meaning |
|---|---|
| Problem importance | Is the addressed problem meaningful to a defined scientific or technical community? |
| Contribution distinctiveness | Is the contribution materially different from verified prior work? |
| Generality | Does the result extend beyond one narrow example, if generality is claimed? |
| Utility | Could the method, dataset, material, algorithm, or insight be useful to others? |
| Mechanistic or explanatory value | Does the work explain why something happens, rather than only report a number? |
| Benchmark relevance | Is the comparison fair, current, and meaningful? |
| Reproducibility potential | Can others realistically verify or reuse the work? |
| Limitation transparency | Are scope and failure conditions honestly stated? |

The report should include an impact narrative:

```text
Potential significance: MODERATE

Reason:
The project addresses a relevant problem and reports an improvement over the supplied
baseline. The evidence currently supports the result for the evaluated dataset, but
external generalization and comparison with two important prior methods are missing.
The likely contribution is useful within the studied setting, while broader field impact
remains uncertain.

Evidence IDs: EVD-001, EVD-004, EVD-009
Literature IDs: LIT-002, LIT-005
Limitations: FIND-014, FIND-018
```

Do not use journal prestige, citation counts, social popularity, or an LLM’s confidence as direct evidence that a new project will have impact. Historical bibliometrics can provide context for a field, but they should never become a score for the individual project’s future success.

## 12. LangGraph node design

The publishability assessment should be a subgraph with these nodes:

```text
load_assessment_packet
  → extract_empirical_features
  → evaluate_blocking_rules
  → parallel_llm_assessments
  → validate_llm_outputs
  → calculate_disagreement
  → aggregate_scores
  → synthesize_readiness_report
  → human_readiness_review
```

The deterministic feature and blocker nodes should run before the LLM. The LLM should see the empirical findings and focus on interpretation, contribution, literature positioning, and missing evidence.

A simplified LangGraph state could be:

```python
class PublishabilityState(TypedDict, total=False):
    project_id: str
    assessment_packet_id: str
    feature_ids: list[str]
    blocker_finding_ids: list[str]
    llm_assessment_ids: Annotated[list[str], operator.add]
    llm_disagreement: float
    dimension_scores: dict[str, float]
    confidence_by_dimension: dict[str, float]
    readiness_status: str
    potential_significance: str
    report_id: str
    human_decision: dict
```

The node should return artifact IDs and compact scores, not entire assessment documents in graph state.

## 13. Pseudocode for the assessment node

```python
def extract_empirical_features(state):
    packet = load_packet(state["assessment_packet_id"])
    features = feature_engine.compute(packet)
    feature_ids = persist_features(features)
    return {"feature_ids": feature_ids}


def evaluate_blocking_rules(state):
    features = load_features(state["feature_ids"])
    findings = policy_engine.evaluate_blockers(features)
    finding_ids = persist_findings(findings)
    return {"blocker_finding_ids": finding_ids}


def run_llm_assessor(state, assessor_id: str):
    packet = load_packet(state["assessment_packet_id"])
    response = llm_gateway.structured_call(
        assessor_id=assessor_id,
        input_packet=packet,
        output_schema=LLMReadinessAssessment,
    )
    validate_evidence_ids(response, packet)
    assessment_id = persist_llm_assessment(response)
    return {"llm_assessment_ids": [assessment_id]}


def aggregate_assessment(state):
    features = load_features(state["feature_ids"])
    llm_assessments = load_llm_assessments(state["llm_assessment_ids"])
    blockers = load_findings(state["blocker_finding_ids"])

    dimension_scores = combine_empirical_and_llm(
        features=features,
        llm_assessments=llm_assessments,
    )
    disagreement = calculate_disagreement(llm_assessments)
    status = apply_readiness_policy(
        dimension_scores=dimension_scores,
        blockers=blockers,
        disagreement=disagreement,
    )

    report_id = persist_readiness_report(
        dimension_scores=dimension_scores,
        llm_disagreement=disagreement,
        status=status,
        feature_ids=state["feature_ids"],
        assessment_ids=state["llm_assessment_ids"],
        blocker_finding_ids=state["blocker_finding_ids"],
    )
    return {
        "dimension_scores": dimension_scores,
        "llm_disagreement": disagreement,
        "readiness_status": status,
        "report_id": report_id,
    }
```

## 14. Calibration and quality control

The node must be calibrated against a small, curated set of real or synthetic-but-expert-authored project cases. Each case should contain expected findings, supported and unsupported claims, domain-specific blockers, and an expert readiness label.

Calibration should measure:

| Measure | Purpose |
|---|---|
| Blocker recall | Does the system catch critical scientific-integrity issues? |
| False-block rate | Does it avoid blocking sound projects for irrelevant reasons? |
| Claim-support agreement | Does the assessment agree with expert evidence judgments? |
| Domain-check coverage | Are required DFT, ML, or quantum checks executed? |
| Abstention quality | Does the system refuse to score when evidence is insufficient? |
| Inter-assessor disagreement | Are LLM outputs stable enough for the current task? |
| Explanation traceability | Does every conclusion cite evidence and findings? |

Optimize first for **blocker recall and traceability**, not for a high average score or fluent prose. A system that misses a fabricated result is worse than one that gives conservative warnings.

## 15. Human approval policy

The system should interrupt for human review when:

- A central contribution is rated `UNCLEAR` or `NOT_SUPPORTED`.
- LLM disagreement exceeds the domain threshold.
- The impact assessor proposes broad generalization not supported by evidence.
- A central claim has only indirect evidence.
- The status is `READY_FOR_MANUSCRIPT` but any high-risk finding remains unresolved.
- The project requires additional calculations or experiments.
- The target journal or article type is uncertain.

The human should be shown the report, claim–evidence table, deterministic feature values, blockers, assessor disagreements, and proposed next actions. The approval decision must be stored with the report version and affected artifact IDs.

## 16. Recommended report format

The report should contain:

1. Assessment status and scope.
2. Project and domain summary.
3. Candidate contribution statements.
4. Dimension scores with empirical/LLM composition.
5. Central claim–evidence matrix.
6. Critical blockers and warnings.
7. Potential significance assessment.
8. Literature-positioning summary.
9. Missing experiments, calculations, controls, or documentation.
10. Reproducibility and consistency results.
11. Confidence and assessor disagreement.
12. Recommended next action.
13. Human approval section.

The recommended next action should be one of:

```text
PROCEED_TO_MANUSCRIPT_PLANNING
DRAFT_WITH_WARNINGS
CREATE_RESEARCH_COMPLETION_PLAN
REQUEST_MORE_PROJECT_INFORMATION
BLOCK_UNTIL_INTEGRITY_ISSUE_RESOLVED
```

## 17. Final recommendation

Implement the publishability node as a **policy-governed evidence assessment**, not as an LLM judge. Let deterministic code calculate coverage, consistency, reproducibility, validation, and domain features. Let LLMs interpret verified evidence, compare contribution candidates with supplied literature, identify plausible weaknesses, and write explanations. Use calibrated disagreement and abstention. Apply hard blockers before score aggregation. Require human approval before the system transitions to manuscript generation.

The most defensible output is not “this project will be published.” It is:

> “Given the artifacts and evidence currently available, the project is **DRAFTABLE_WITH_WARNINGS**. The central result is supported by artifacts EVD-001 through EVD-006, but external validation and convergence evidence are missing. The potential significance is moderate and may increase if the proposed completion tasks are completed. Human approval is required before manuscript planning.”

That output is useful, auditable, scientifically conservative, and implementable in LangGraph.
