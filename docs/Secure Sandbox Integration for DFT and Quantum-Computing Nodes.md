# Secure Sandbox Integration for DFT and Quantum-Computing Nodes

## 1. Architectural rule

LangGraph should orchestrate scientific jobs but should not execute arbitrary DFT, quantum-computing, shell, notebook, or user-provided code inside a graph node. A graph node should create a validated `AnalysisJobSpec`, request human approval when required, submit the specification to a separate bounded job runner, and later ingest immutable result artifacts.

```text
LangGraph node
  → validate typed job specification
  → policy check
  → human approval if required
  → sandbox/job runner
  → read-only input mount + isolated output mount
  → bounded scientific execution
  → artifact registration and checksums
  → result summary returned to LangGraph
```

The sandbox is a security boundary, not merely a Conda environment. Conda isolates Python packages but does not by itself provide sufficient protection against arbitrary code, unsafe subprocesses, filesystem writes, network access, resource exhaustion, or malicious input files.

## 2. Three execution tiers

| Tier | Use | Default permissions |
|---|---|---|
| Tier 0: Static inspection | File parsing, metadata extraction, notebook/code inspection, figure inspection | No code execution, read-only files |
| Tier 1: Bounded analysis | Trusted parsers, deterministic calculations, approved Python analysis modules | Read-only inputs, private output directory, no network, CPU/memory/time limits |
| Tier 2: Heavy scientific job | DFT, quantum chemistry, large ML training, simulation, or hardware emulation | Dedicated worker/container/VM, strict profile, resource reservation, no inherited host credentials |

The first release should support Tier 0 and a small Tier 1. DFT and heavyweight quantum calculations should be added through Tier 2 profiles after the artifact and provenance system is stable.

## 3. Job specification

A job must be declarative. The LLM or supervisor may select an approved profile, but it must not construct arbitrary shell commands.

```yaml
job_id: JOB-0001
project_id: PROJECT-0001
job_type: DFT
profile_id: dft.vasp.convergence.v1
input_artifact_ids:
  - ART-structure-0001
  - ART-pseudopotential-0001
parameters:
  cutoff_eV: 520
  kpoint_mesh: [6, 6, 6]
  convergence_energy_eV: 0.00001
resource_limits:
  cpu_cores: 8
  memory_mb: 32768
  wall_time_seconds: 86400
  max_output_bytes: 10737418240
network:
  enabled: false
  allowlist: []
execution:
  environment_lock_hash: sha256:...
  code_revision: git:...
  command_profile: dft.vasp.convergence.v1
approval:
  required: true
```

The job runner should reject unknown profile IDs, unexpected parameters, paths, executable names, environment identifiers, and network requests.

## 4. Input and output handling

Inputs should be selected by artifact ID, not by arbitrary paths supplied inside an LLM response. The artifact service resolves approved IDs to immutable files and mounts them read-only. The worker receives a generated input manifest containing checksums and expected media types.

Outputs should be written to a newly created job directory. After execution, the runner should calculate checksums, validate expected file types, record exit status and resource usage, and register outputs as derived artifacts. The runner must not replace the source artifact or modify the original project folder.

```text
managed_project/
├── raw_readonly/          # never writable by job
├── job_inputs/JOB-0001/   # read-only materialized inputs
├── job_outputs/JOB-0001/  # writable only for this job
└── artifacts/             # immutable committed outputs
```

## 5. DFT-specific controls

DFT profiles should be explicit and domain-aware. The runner should validate structural files, pseudopotentials, basis or functional selection, cutoff energy, k-point mesh, spin settings, charge, cell, convergence thresholds, relaxation criteria, and expected output files before execution.

For VASP, Quantum ESPRESSO, Gaussian, CP2K, or other codes, each executable should have its own versioned profile. The profile should define allowed input files, supported parameters, command templates maintained by developers, expected output parsers, and validation rules. The LLM should choose from a profile and parameter schema; it should not write an unrestricted command line.

A DFT execution should produce at least:

| Artifact | Purpose |
|---|---|
| Input manifest | Exact structure, pseudopotential, functional, parameters, and checksums |
| Environment manifest | Code version, compiler, libraries, and environment lock hash |
| Execution log | Start/end time, exit code, resource usage, stdout/stderr |
| Raw output | Original code output, immutable after commit |
| Parsed result | Energies, forces, stress, band data, phonons, or other typed values |
| Validation report | Convergence, parsing, unit, and expected-output checks |
| Provenance record | Parent artifacts, job specification, code revision, and human approvals |

A parsed numerical result must never be accepted without preserving the original output artifact and parser version.

## 6. Quantum-computing controls

Quantum profiles should distinguish simulation from hardware execution. A simulator job may be isolated locally, while hardware or cloud execution requires a separate allowlisted connector and an additional human approval gate.

A quantum job profile should validate the Hamiltonian or circuit, qubit count, mapping, active space, ansatz, optimizer, initialization, shots, noise model, simulator/backend, seed, maximum iterations, convergence criteria, and expected result schema.

Hardware execution should be disabled by default. If enabled, the job runner must use a dedicated connector with narrowly scoped credentials, never expose credentials to the LLM or arbitrary subprocesses, record the backend and job ID, and require explicit approval before submission. The system should treat hardware results as external evidence with provider metadata and retrieval timestamps.

## 7. Container, VM, or scheduler choices

| Deployment context | Recommended boundary |
|---|---|
| Local prototype | Dedicated subprocess plus restricted working directory and OS resource limits; only for trusted profiles |
| Linux workstation | Rootless container with dropped capabilities, read-only root filesystem, seccomp/AppArmor or equivalent, no network, and explicit mounts |
| Heavy DFT workstation/server | Dedicated container or VM per job profile, optionally submitted to Slurm or another scheduler |
| Sensitive or untrusted project code | Disposable VM or stronger isolation; do not execute directly on the host |
| Cloud quantum hardware | Separate connector service with allowlisted API, short-lived credentials, approval gate, and audit record |

Docker is not automatically a complete security boundary for hostile workloads. For untrusted code or high-consequence environments, use a stronger isolation boundary such as a disposable VM or a hardened sandbox runtime. The system should begin with trusted, profile-driven jobs and expand only after security testing.

## 8. Required OS/container controls

A Tier 2 runner should apply the following controls:

| Control | Requirement |
|---|---|
| User identity | Run as non-root, dedicated UID/GID, no host user credentials |
| Filesystem | Read-only root filesystem; only job output directory writable |
| Mounts | Explicit input mounts, read-only; no host home directory or Docker socket |
| Network | Disabled by default; allowlist only for approved connectors |
| Linux capabilities | Drop all unnecessary capabilities |
| System calls | Apply seccomp and mandatory access-control profiles where available |
| Resources | CPU, memory, process count, wall time, file size, and disk quotas |
| Environment | Pinned image/Conda lock hash; no unapproved package installation |
| Secrets | Never expose secrets as files or environment variables to analysis code unless required and narrowly scoped |
| Cleanup | Destroy temporary workspace after artifact commit, retaining only logs and registered outputs |
| Monitoring | Capture exit code, signals, resource usage, and timeout/kill reason |

## 9. LangGraph integration pattern

The LangGraph node should do policy and submission orchestration only.

```python
from dataclasses import asdict


def create_dft_job(state):
    spec = AnalysisJobSpec(
        job_id=new_id("JOB"),
        project_id=state["context"]["project_id"],
        job_type="DFT",
        input_artifact_ids=tuple(state["selected_input_artifact_ids"]),
        command_profile="dft.vasp.convergence.v1",
        environment_lock_hash=state["environment_lock_hash"],
        cpu_limit=8,
        memory_mb=32_000,
        wall_time_seconds=86_400,
        network_enabled=False,
    )

    policy_result = policy_engine.validate_job(spec)
    if not policy_result.allowed:
        return {"status": "BLOCKED", "finding_ids": policy_result.finding_ids}

    job_spec_id = artifact_store.commit_json(
        kind="analysis_job_spec",
        payload=asdict(spec),
        parent_ids=list(spec.input_artifact_ids),
    )
    return {"job_spec_id": job_spec_id, "status": "AWAITING_APPROVAL"}
```

After approval, a separate submission node calls a job-runner API. It should receive a job ID, not wait indefinitely inside the LangGraph process. A polling or callback mechanism should later update the run with `SUCCEEDED`, `FAILED`, `TIMED_OUT`, or `CANCELLED` and attach output artifact IDs.

## 10. Job-runner API contract

```http
POST /v1/jobs
Content-Type: application/json

{
  "job_spec_id": "ART-JOB-0001",
  "profile_id": "dft.vasp.convergence.v1",
  "input_artifact_ids": ["ART-0001"],
  "resource_limits": {
    "cpu_cores": 8,
    "memory_mb": 32000,
    "wall_time_seconds": 86400,
    "max_output_bytes": 10737418240
  }
}
```

Response:

```json
{
  "job_id": "JOB-0001",
  "status": "QUEUED",
  "runner_revision": "runner-git-revision",
  "profile_version": "dft.vasp.convergence.v1"
}
```

Status response:

```json
{
  "job_id": "JOB-0001",
  "status": "SUCCEEDED",
  "exit_code": 0,
  "output_artifact_ids": ["ART-OUT-0001", "ART-PARSED-0001"],
  "execution_log_artifact_id": "ART-LOG-0001",
  "resource_usage": {
    "cpu_seconds": 12345,
    "peak_memory_mb": 12000,
    "wall_time_seconds": 2300
  }
}
```

The runner must validate the job specification again. Never rely on the LangGraph node as the only policy enforcement point.

## 11. Failure and recovery behavior

Scientific jobs fail frequently for legitimate reasons: convergence failure, invalid input, missing pseudopotential, insufficient memory, scheduler cancellation, parser errors, or unavailable hardware. These outcomes must be represented explicitly.

| Failure | LangGraph action |
|---|---|
| Invalid job specification | `BLOCKED`, create remediation finding |
| Sandbox policy violation | `BLOCKED`, do not retry automatically |
| Transient runner unavailable | Retry submission with same idempotency key |
| Scientific non-convergence | `FAILED` or `COMPLETED_WITH_WARNING`, no invented result |
| Timeout | `TIMED_OUT`, preserve partial logs if valid, require review |
| Parser failure | Keep raw output, create parser finding, do not publish parsed values |
| Hardware/API failure | Record provider job state and retry only under connector policy |
| Output checksum failure | Reject output artifact and block downstream use |

The graph should separate **retryable infrastructure failures** from **scientific failures**. Retrying a non-converged DFT calculation without changing the specification is usually not useful and may waste resources.

## 12. Minimum security acceptance tests

Before enabling DFT or quantum execution, verify that:

1. A job cannot write to the original project folder.
2. A job cannot read files outside its explicit input artifact mounts.
3. A job cannot access host credentials, Docker sockets, SSH keys, or unrelated environment variables.
4. Network access is disabled unless the profile explicitly permits an allowlisted connector.
5. CPU, memory, process, wall-time, and output-size limits terminate abusive jobs.
6. An unknown executable or parameter is rejected by the profile validator.
7. A job cannot install packages or modify the pinned environment.
8. The raw output is preserved before parsing.
9. A parser cannot replace an existing artifact with different content.
10. Resubmission uses an idempotency key and does not duplicate committed outputs.
11. A human approval is required before hardware/cloud execution.
12. A failed or killed job produces an explicit finding and never a successful result.

## 13. Recommended implementation sequence

Start with a fake job runner that accepts approved specifications and returns deterministic fixture artifacts. Use it to test LangGraph routing, interrupts, persistence, idempotency, and provenance before running real scientific software.

Then implement static DFT and quantum parsers, followed by a small trusted analysis profile in a restricted environment. Only after these controls pass should you add heavy DFT executables, large simulations, or cloud quantum hardware connectors.

> **Do not connect LangGraph directly to `subprocess.run()` with an LLM-generated command.** Use a typed profile registry, validated parameters, immutable artifact IDs, a bounded runner, and a second policy check inside the runner.
