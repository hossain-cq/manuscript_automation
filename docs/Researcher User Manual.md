# Researcher User Manual

## Scientific Manuscript System CLI

**CLI:** `manuscriptctl`  
**Audience:** Researchers working on DFT/materials, AI/ML, quantum-computing, and related computational projects  
**Author:** Manus AI

## 1. What this tool does

The manuscript system turns a completed or partially completed research project into a controlled publication-readiness assessment and, after approval, a draft manuscript and review package. You provide a project folder, select a scientific domain and target journal, and describe what you want assessed.

The system does not assume that a project is publishable because it contains code, data, or figures. It reconstructs the project, checks evidence and provenance, identifies missing analyses, assesses literature positioning, prepares a manuscript plan, writes evidence-linked sections, simulates peer review, and pauses for researcher approval at important decisions.

> The CLI starts and manages jobs. It does not replace scientific judgment, domain review, peer review, or journal editorial decisions.

## 2. Operating modes

| Mode | What it does | Recommended use |
|---|---|---|
| `assessment` | Intake, project understanding, evidence audit, literature grounding, novelty comparison, and readiness report | First run for an unfamiliar project |
| `manuscript` | Drafts sections from an approved assessment and evidence packet | After assessment has been reviewed |
| `full` | Runs assessment, planning, drafting, review simulation, and revision workflow | Mature projects with organized artifacts |

For a first project, use `assessment`. Do not begin with `full` until you have confirmed that the system can correctly interpret the project folder.

## 3. Installation

Create and activate a virtual environment in the repository:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r deploy/requirements.lock
chmod +x manuscriptctl.py
```

If the repository uses a fully hash-locked dependency file, install with:

```bash
python -m pip install --require-hashes -r deploy/requirements.lock
```

The CLI requires a reachable control-plane API. For local Docker Compose deployment:

```bash
docker compose -f deploy/docker-compose.yml up -d postgres runner api
```

Set the endpoint and bearer token if authentication is enabled:

```bash
export MANUSCRIPT_API_URL=http://127.0.0.1:8080
export MANUSCRIPT_API_TOKEN='your-short-lived-token'
```

Never place API keys in the project folder or commit them to Git.

## 4. Check service health

Before uploading a project, verify that the control plane is ready:

```bash
./manuscriptctl.py health
```

A healthy response should indicate that the API and its required database/checkpoint dependencies are available. If the command fails, inspect the API logs before trying to start a project job.

## 5. Prepare a project folder

The selected folder should contain the relevant inputs, source code, data, outputs, logs, figures, environment information, and documentation. The system can work with an incomplete folder, but it should report the incompleteness rather than conceal it.

A useful project layout is:

```text
my-project/
├── README.md
├── run_manifest.json
├── environment.yml or requirements.lock
├── src/ or notebooks/
├── inputs/
├── raw_data/
├── derived_data/
├── results/
├── figures/
├── logs/
└── references/
```

Before submission, remove secrets and personal data. The CLI excludes common credential files and hidden development directories when creating its filtered archive, but you remain responsible for reviewing the selected folder.

The CLI does not follow symlinks outside the project boundary, refuses filesystem roots, records file checksums, skips common secret patterns, and rejects individual files over the configured size limit.

## 6. Start an assessment job

For a DFT/materials project targeting Nature:

```bash
./manuscriptctl.py start \
  --project /work/projects/my-dft-project \
  --domain dft_materials \
  --journal nature \
  --mode assessment \
  --instruction "Inspect the project, assess publication readiness, identify missing convergence and validation evidence, and recommend a manuscript structure."
```

For an AI/ML project targeting Physical Review Letters:

```bash
./manuscriptctl.py start \
  --project /work/projects/my-ml-project \
  --domain ai_ml \
  --journal physical_review_letters \
  --mode assessment \
  --instruction "Assess the scientific contribution, baseline quality, data leakage risks, uncertainty, generalization evidence, and suitability for a concise Letter."
```

For a mature project that has already passed internal review:

```bash
./manuscriptctl.py start \
  --project /work/projects/my-project \
  --domain dft_materials \
  --journal nature \
  --mode full \
  --name "Convergence-controlled materials study"
```

The command creates a filtered archive and sends it to the control plane. The API should return a run identifier, for example `RUN-abc123`. Save this identifier.

## 7. Preview the upload

Use `--dry-run` to create and validate the archive request without starting a server-side run:

```bash
./manuscriptctl.py start \
  --project /work/projects/my-dft-project \
  --domain dft_materials \
  --journal nature \
  --mode assessment \
  --dry-run
```

A production API should return the manifest preview without scheduling downstream analysis. Confirm that the file count, paths, sizes, and excluded patterns are appropriate.

## 8. Monitor a run

Check the current state:

```bash
./manuscriptctl.py status RUN-abc123
```

Follow structured run events:

```bash
./manuscriptctl.py logs RUN-abc123 --follow
```

Stop following with `Ctrl+C`; this does not cancel the server-side run. You can control the polling interval:

```bash
./manuscriptctl.py logs RUN-abc123 --follow --interval 10
```

Typical stages include intake, manifest creation, domain classification, artifact indexing, data sufficiency, literature grounding, novelty comparison, publishability assessment, human approval, manuscript drafting, figure generation, review simulation, revision, and release packaging.

## 9. Human approvals and resume behavior

The system deliberately pauses at important approval gates. A status or event response should identify the approval type and the material that requires review.

Approve a gate:

```bash
./manuscriptctl.py approve RUN-abc123 \
  --decision approve \
  --reason "I reviewed the project inventory, evidence matrix, and missing-analysis report."
```

Request changes:

```bash
./manuscriptctl.py approve RUN-abc123 \
  --decision request_changes \
  --reason "Add the k-point convergence analysis before drafting the central stability claim."
```

Reject a gate:

```bash
./manuscriptctl.py approve RUN-abc123 \
  --decision reject \
  --reason "The current artifact set does not correspond to the described project."
```

Resume an interrupted LangGraph run:

```bash
./manuscriptctl.py resume RUN-abc123 \
  --decision APPROVE_REVISION_PLAN \
  --reason "The accepted comments and proposed changes are scientifically appropriate."
```

The control plane must associate the resume operation with the same durable LangGraph thread. Do not create a new run merely because the first run is waiting for approval.

## 10. How to read the assessment

The assessment should distinguish:

| Status | Meaning |
|---|---|
| Ready for manuscript drafting | Current approved evidence satisfies the configured readiness checks, subject to human confirmation |
| Draftable with warnings | A draft can be prepared, but limitations or missing non-critical evidence must be disclosed |
| Needs additional analysis | Existing data may be useful, but required checks such as convergence, baselines, uncertainty, or validation are missing |
| Needs additional experiment/calculation | The central claim cannot be responsibly supported by wording changes alone |
| Contribution unclear | The project is understood, but novelty or scientific importance is not yet defensible |
| Insufficient evidence to assess | The folder lacks enough interpretable artifacts or metadata |

A high score should never override a critical blocker. For example, missing data-leakage analysis can block an AI/ML generalization claim even if the accuracy metric is high.

## 11. DFT projects

For DFT/materials work, ensure that the folder includes computational settings and relevant output tables. The system looks for software/version, functional, pseudopotential or basis, cutoff, k-point mesh, convergence criteria, charge/spin state, structure identity, relaxation details, reference states, units, and comparison systems.

The system may recommend convergence plots, formation-energy comparisons, band structures, density of states, phonon analyses, structural comparisons, or sensitivity plots. These figures should be generated from registered data artifacts rather than screenshots or manually edited numbers.

The system distinguishes numerical convergence from physical correctness. A converged calculation is not automatically a proof of stability, experimental feasibility, or universal transferability.

## 12. AI/ML projects

For AI/ML work, provide dataset provenance, preprocessing, split definitions, duplicate/leakage checks, seeds, architecture, hyperparameters, baselines, metric definitions, uncertainty or repeated-run information, error analysis, and external or out-of-distribution evaluation where relevant.

The system may recommend parity plots, confusion matrices, ROC/PR curves, learning curves, error distributions, calibration plots, ablations, baseline comparisons, and subgroup analysis. It should not generate a generalization claim from a single aggregate test score.

## 13. Quantum-computing projects

For quantum-computing work, the project should record the Hamiltonian or problem definition, mapping, ansatz, active space, optimizer, shot count, noise model, backend or simulator, initialization, convergence, statistical uncertainty, classical baselines, and hardware limitations.

Hardware or cloud execution must require an explicit approval gate. The CLI only starts the workflow; the secure runner controls whether a job profile may access a simulator, local hardware, or an external provider.

## 14. Manuscript generation

After approving the assessment, start or resume the manuscript stage according to the API’s workflow controls. The section writer generates versioned blocks for title, abstract, introduction, methods, results, discussion, conclusion, data availability, code availability, and references as required by the journal profile.

Every block should display its claim IDs, evidence IDs, literature IDs, model ID, prompt version, and warnings. Review the evidence links before accepting a block. If a block says that additional work is required, do not force the system to write a stronger claim.

## 15. Peer-review simulation

The system can simulate scientific-expert, critical, statistical/computational, journal, and hostile reviewer perspectives. These are model-generated pre-submission critiques, not real peer review.

Read each comment’s severity, category, affected block, affected claim, evidence IDs, validity, and proposed action. A valid request for a new calculation should become a research-completion task. It should not be “resolved” by replacing a strong sentence with a vague sentence while keeping the unsupported conclusion.

Approve the revision plan before changes are made. After revision, inspect the numeric-token change report, evidence-link validation, contradiction scan, and response-to-reviewers document.

## 16. Export a release package

Export should be enabled only after the final human approval gate:

```bash
./manuscriptctl.py export RUN-abc123 --output ./release/RUN-abc123-release.zip
```

A release package should contain the manuscript, figures, tables, supplementary material, bibliography, claim/evidence map, provenance manifest, review reports, revision history, response-to-reviewers document, environment information, and approval audit.

## 17. Cancel a run

Cancel a run when it is consuming resources incorrectly, using the wrong project, or no longer needed:

```bash
./manuscriptctl.py cancel RUN-abc123 --reason "Wrong project folder selected; restart with corrected intake."
```

Cancellation should be recorded as an audit event. Long-running scientific jobs should also be cancelled in the runner or scheduler, not merely hidden from the CLI.

## 18. Troubleshooting

| Symptom | Likely cause | Action |
|---|---|---|
| Health check fails | API, database, or runner is unavailable | Inspect Compose/Kubernetes status and service logs |
| Upload rejected | File too large, invalid path, or archive policy violation | Use `--max-file-mb`, remove unsuitable artifacts, and retry after reviewing the folder |
| No DFT readiness | Missing settings or convergence evidence | Add the run manifest and required convergence outputs |
| AI/ML claim blocked | Missing leakage check, baseline, or split definition | Run and register the missing analysis |
| Literature stage has no result | Provider unavailable or query too narrow | Inspect provider events; do not interpret absence as novelty |
| Draft pauses | Human approval is required | Inspect status/events, then use `approve` or `resume` |
| Revision rejected | New unsupported number, citation, or claim detected | Review the verification findings and update the evidence packet |
| Export unavailable | Final release approval is missing | Complete the review and approval gates |

## 19. Security rules for researchers

Do not upload secrets, SSH keys, cloud credentials, private tokens, personal data, or unrelated confidential projects. Treat notebooks and scripts as untrusted inputs. Do not assume that a Conda environment protects against arbitrary execution. The server-side runner must enforce isolation, and the CLI must not be used as a way to execute project scripts locally.

When a job requests permission to run a calculation, inspect the job profile, input artifact IDs, output limits, software image, network policy, and expected cost before approving it. For external quantum hardware or cloud computation, verify the provider, budget, backend, shot count, and data policy.

## 20. Reproducibility checklist

Before release, confirm that the project has a stable project identifier, source checksums, code revision, environment lock, run parameters, raw and derived artifact lineage, literature metadata, plot specifications, manuscript block versions, reviewer reports, revision decisions, and human approvals.

## 21. CLI reference

```text
manuscriptctl health
manuscriptctl start --project PATH --domain DOMAIN --journal JOURNAL [OPTIONS]
manuscriptctl status RUN_ID
manuscriptctl logs RUN_ID [--follow] [--interval SECONDS]
manuscriptctl approve RUN_ID --decision {approve,reject,request_changes} --reason TEXT
manuscriptctl resume RUN_ID --decision DECISION [--reason TEXT]
manuscriptctl cancel RUN_ID --reason TEXT
manuscriptctl export RUN_ID --output PATH
```

Global options are available on each command:

```text
--url URL       Control-plane API URL
--token TOKEN   Bearer token; otherwise MANUSCRIPT_API_TOKEN
```

## 22. Responsible interpretation

The system is designed to make scientific work more organized, auditable, and reviewable. It cannot establish truth by language generation, guarantee novelty, predict acceptance, or replace a qualified researcher. The final responsibility for claims, data, figures, citations, authorship, ethics, and submission remains with the research team.
