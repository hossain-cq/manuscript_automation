# Automated Data and Plot Sufficiency Checks

## DFT and AI/ML implementation specification

## 1. Design principle

The system must distinguish three separate questions:

1. **Data sufficiency:** Are the required source data, outputs, metadata, controls, baselines, and uncertainty records present?
2. **Plot sufficiency:** Are the required plots or tables available to support the planned claims and manuscript sections?
3. **Figure quality:** Are the figures technically correct, readable, traceable, and suitable for a manuscript?

A project can have many plots but still lack sufficient evidence. Conversely, it may have enough data but need new figures. The system should therefore report data coverage, evidence coverage, and figure readiness separately.

> A generated plot is not automatically evidence. It is a derived artifact whose source data, transformation, units, code, and claim relationship must be recorded.

## 2. Recommended LangGraph workflow

```text
approved claims and domain profile
  → load data/figure manifest
  → deterministic data integrity checks
  → domain-specific sufficiency checks
  → map claims to required plot specifications
  → identify missing or weak plots
  → generate only approved deterministic figures
  → validate figure files and source lineage
  → create caption proposals
  → human figure review
  → approve/reject/regenerate
```

LLMs may help infer candidate plot requirements from approved claims, but they must not decide that a plot proves a claim. Plot requirements should be confirmed by the researcher or a domain profile.

## 3. Data manifest

Every data asset should be registered before analysis.

```yaml
data_asset:
  asset_id: ART-DATA-0001
  path: "/managed/project/derived/results.csv"
  kind: CSV
  checksum: "sha256:..."
  created_by_run_id: RUN-0001
  parent_artifact_ids:
    - ART-RAW-0001
  columns:
    - cutoff_eV
    - total_energy_eV
  units:
    cutoff_eV: eV
    total_energy_eV: eV
  row_count: 12
  provenance:
    code_revision: "git:..."
    environment_lock_hash: "sha256:..."
    generation_command: "approved job profile, not arbitrary shell"
```

The source data must remain immutable. Derived tables may be generated, but they must reference their parent assets and preserve transformation metadata.

## 4. Sufficiency report

```yaml
sufficiency_report:
  report_id: SUFF-0001
  project_id: PROJECT-0001
  profile_id: dft_materials.v1
  status: SUFFICIENT_WITH_WARNINGS
  data_coverage_score: 0.82
  evidence_coverage_score: 0.68
  plot_readiness_score: 0.75
  checks:
    - CHECK-0001
  missing_requirements:
    - "No independent reference calculation"
  critical_blockers: []
  recommended_actions:
    - "Add k-point convergence plot for central formation-energy claim"
  human_review_required: true
```

The status should be `SUFFICIENT`, `SUFFICIENT_WITH_WARNINGS`, `INSUFFICIENT`, or `NOT_ASSESSABLE`. These are readiness statuses, not publication guarantees.

## 5. Universal deterministic checks

The following checks should run for every project and profile:

| Check | Required behavior |
|---|---|
| File existence | Verify that registered assets exist at the managed path |
| Checksum | Detect changes after registration |
| Format readability | Parse supported CSV, TSV, Excel, JSON, NPY, and image files |
| Empty data | Flag empty tables or zero-sized arrays |
| Non-finite values | Detect NaN, positive infinity, and negative infinity in numeric fields |
| Duplicate rows | Report duplicate data rows where relevant |
| Column schema | Verify required columns and types |
| Units | Require units for physical quantities and metrics where applicable |
| Source lineage | Link every derived artifact to parent assets |
| Reproducibility metadata | Check code revision, environment, parameters, and run identifier |
| Plot-source mapping | Confirm each figure references source data artifacts |
| Claim mapping | Confirm central claims reference evidence and, where required, figures/tables |

A failed universal check should create a structured finding rather than silently excluding the asset.

## 6. DFT profile

The DFT profile should be configurable by code family, such as VASP, Quantum ESPRESSO, CP2K, Gaussian, or another approved profile. The profile should not assume that every project needs every possible output.

### 6.1 Minimum DFT metadata

| Metadata | Why it matters |
|---|---|
| Software and version | Reproducibility and parser interpretation |
| Functional | Scientific model definition |
| Pseudopotential or basis | Reproducibility and comparability |
| Cutoff | Numerical convergence |
| k-point mesh | Brillouin-zone sampling |
| Structure/molecule identifier | Input identity |
| Charge and spin state | Physical model definition |
| Relaxation criteria | Structural validity |
| Convergence criteria | Numerical reliability |
| Temperature or correction model | Thermodynamic interpretation where relevant |
| Reference states | Formation energy and relative-property validity |

### 6.2 DFT sufficiency checks

The system should create checks based on the claim type.

| Claim type | Expected evidence or plots |
|---|---|
| Total-energy comparison | Consistent structures, units, reference states, and convergence evidence |
| Formation energy | Reference chemical potentials, stoichiometry, units, and competing phases where relevant |
| Structural stability | Relaxation convergence, forces/stress, and structural comparison |
| Electronic structure | Band/DOS data, path definition, Fermi-level convention, and figure source data |
| Phonon or dynamical stability | Phonon output, q-point path/mesh, imaginary-mode interpretation, and convergence context |
| Battery or energy material claim | Voltage/energy definitions, reference states, composition, and comparison systems |
| Method comparison | Same or explicitly controlled settings across methods |

The system should never mark DFT data sufficient solely because a file named `CONVERGENCE` or `final_results.csv` exists. It must inspect the schema, values, metadata, and relation to the claim.

### 6.3 DFT figure requirements

Typical profile-driven plots include cutoff convergence, k-point convergence, energy/volume curves, band structures, density of states, phonon dispersion, formation-energy comparisons, structural diagrams, and parameter-sensitivity plots. Each plot must have a declared source artifact, x/y columns or arrays, units, claim relationship, and expected interpretation.

## 7. AI/ML profile

The AI/ML profile must focus on experimental design and generalization rather than only visual quality.

### 7.1 Minimum AI/ML metadata

| Metadata | Why it matters |
|---|---|
| Dataset source and version | Provenance and reproducibility |
| Preprocessing | Reconstructing transformations and avoiding leakage |
| Split definition | Validity of evaluation |
| Random seed(s) | Repeatability |
| Model architecture and parameters | Method completeness |
| Baseline models | Meaningful comparison |
| Metrics and definitions | Correct interpretation |
| Training environment | Reproducibility |
| Hyperparameters | Reproduction and sensitivity |
| Evaluation protocol | Reliability of reported results |

### 7.2 AI/ML sufficiency checks

| Claim type | Expected evidence or plots |
|---|---|
| Performance improvement | Baseline comparison, uncertainty/repeated runs, same evaluation protocol |
| Generalization | Independent or appropriately separated test set, leakage checks, subgroup or out-of-distribution analysis |
| Model mechanism | Ablation, feature importance limitations, error analysis, and controlled comparisons |
| Training stability | Learning curves across runs or seeds, convergence behavior, and failed-run accounting |
| Classification quality | Confusion matrix, per-class metrics, class balance, ROC/PR where applicable |
| Regression quality | Parity plot, residual/error distribution, calibration or uncertainty where relevant |
| Deployment/application value | Operational metric, threshold analysis, failure cases, and domain baseline |

A high aggregate metric must not compensate for data leakage, missing baselines, or an invalid split. Those findings should block the associated claim.

## 8. Plot requirement registry

The figure agent should work from explicit `PlotRequirement` objects.

```yaml
plot_requirement:
  requirement_id: PLOT-0001
  claim_id: CLM-0001
  plot_type: PARITY
  title: "Predicted versus reference energy"
  source_asset_ids:
    - ART-PRED-0001
  required_columns:
    - reference
    - prediction
  units:
    reference: eV
    prediction: eV
  required_checks:
    - finite_values
    - units_present
    - same_sample_ids
    - diagonal_reference_line
  importance: CENTRAL
  journal_role: main_figure
```

A plot requirement may be generated from a domain profile, a manuscript template, or an LLM proposal. In the latter case, the researcher should approve it before generation.

## 9. Automated figure generation

Figures should be generated by deterministic Python plotting code from approved plot specifications. Do not ask an LLM to draw scientific plots directly. The LLM may draft captions or propose a plot type, but numeric plotting should use code and registered data artifacts.

The generation function must:

1. Resolve input artifact IDs through the artifact repository.
2. Verify source checksums.
3. Validate required columns and units.
4. Apply a versioned plotting style.
5. Generate the figure in a managed output directory.
6. Record source artifact IDs, source checksums, plot specification hash, code revision, and environment lock hash.
7. Run deterministic file checks.
8. Create a caption proposal that does not claim more than the data support.
9. Mark the figure as `PROPOSED` until a human approves it.

### 9.1 Figure provenance

```yaml
figure_provenance:
  figure_id: FIG-0001
  plot_requirement_id: PLOT-0001
  source_asset_ids:
    - ART-PRED-0001
  source_checksums:
    ART-PRED-0001: "sha256:..."
  generation_code_revision: "git:abc123"
  plotting_environment_lock_hash: "sha256:..."
  plot_spec_hash: "sha256:..."
  generated_at: "timestamp"
  claim_ids:
    - CLM-0001
  status: PROPOSED
```

## 10. Figure validation

Figure validation should have two layers.

### 10.1 Deterministic technical validation

Check file existence, dimensions, resolution, format, empty output, axis-label presence, legend consistency, units, finite data, source checksums, and cross-reference metadata.

### 10.2 Scientific review

A human or domain reviewer should inspect whether the figure supports the stated claim, whether the selected range is misleading, whether uncertainty is visible, whether comparisons are fair, and whether the caption is accurate. An image-quality model may help identify clipping or unreadable text, but it must not approve scientific validity alone.

## 11. Caption generation

The caption agent should receive the plot requirement, source evidence, deterministic checks, and claim text. It should return structured output:

```python
class CaptionDraft(BaseModel):
    figure_id: str
    caption: str
    variables_and_units: dict[str, str]
    data_scope: str
    limitations: list[str]
    evidence_ids: list[str]
    overclaim_warning: bool
```

The caption should describe what is shown, identify variables and units, state the sample or calculation scope, and avoid unsupported conclusions. A caption should not say “demonstrates superior performance” unless the relevant comparison and uncertainty checks support that wording.

## 12. LangGraph figure workflow

```text
approved claims
  → load profile
  → load data manifest
  → run universal checks
  → run DFT or AI/ML checks
  → calculate sufficiency report
  → create/confirm plot requirements
  → generate deterministic figures
  → validate figure files and provenance
  → generate caption proposals
  → human figure review
  → approve, reject, or regenerate
```

The graph should route differently based on sufficiency:

| Sufficiency status | Figure behavior |
|---|---|
| `SUFFICIENT` | Generate approved required figures and proceed to review |
| `SUFFICIENT_WITH_WARNINGS` | Generate figures but attach warnings and require review |
| `INSUFFICIENT` | Generate diagnostic figures only; do not present the project as manuscript-ready |
| `NOT_ASSESSABLE` | Request missing files or metadata before generation |

## 13. Plot sufficiency versus plot quality

These scores should remain separate.

```text
plot_requirement_coverage =
    approved_requirements_with_valid_source_data / total_approved_requirements

plot_technical_quality =
    passed_file_and_render_checks / applicable_render_checks

plot_evidence_coverage =
    figures_linked_to_supported_claims / central_claims_requiring_figures
```

A project can have high technical quality but low evidence coverage if its plots are polished but do not address the central claims. The system should report this explicitly.

## 14. Recommended plot sets

### DFT and materials

A common starting set may include a convergence plot, a primary result comparison, and one mechanistic or structural visualization. Additional figures should be added only when they answer a defined scientific question.

### AI/ML

A common starting set may include a baseline comparison, parity or confusion analysis, learning curves, error distribution, and an appropriate generalization or calibration view. The exact set depends on the task and claim type.

These are profile defaults, not universal publication requirements. The researcher and target-journal profile should determine the final set.

## 15. Security and reproducibility

The figure node should read registered artifacts and must not execute arbitrary project code. If data transformation is required, use approved deterministic transformation functions with versioned implementations. Scientific simulations, DFT calculations, and model retraining must occur in the separate bounded scientific job runner described in the sandbox specification.

Every generated figure must be reproducible from its plot specification, source artifacts, code revision, environment lock, and style version. If any source checksum changes, the figure should become stale and require regeneration.

## 16. Tests

The figure and sufficiency subsystem should include:

| Test | Expected behavior |
|---|---|
| Missing column | Figure generation fails with a structured finding |
| NaN or infinity | Figure generation refuses the dataset |
| Changed source checksum | Existing figure marked stale |
| Missing units | Warning or blocker according to profile |
| Empty table | Sufficiency fails |
| Missing DFT convergence | Central convergence claim receives a blocker or high warning |
| AI/ML leakage unknown | Generalization claim is not marked ready |
| Baseline missing | Improvement claim is blocked or downgraded |
| Figure orphaned | Figure is flagged if not linked to a manuscript block |
| Claim without figure | Warning when profile marks a figure as required |
| Caption overclaim | Human review required |
| Repeated generation | Same inputs/specification produce same figure checksum where deterministic |

## 17. Final recommendation

Implement literature grounding and figure sufficiency as separate LangGraph subgraphs that share the same claim, evidence, artifact, provenance, and human-decision records. Use LLMs for query expansion, semantic comparison, requirement proposals, and caption drafting. Use deterministic code for metadata verification, data checks, numerical transformations, plotting, checksums, and profile-based sufficiency decisions.

The system should never conclude that a project is complete because it has many figures. It should conclude only that the currently available evidence satisfies a documented set of domain and claim-specific requirements, with limitations and human approval recorded.
