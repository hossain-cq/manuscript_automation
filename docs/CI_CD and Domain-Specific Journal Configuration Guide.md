# CI/CD and Domain-Specific Journal Configuration Guide

## 1. Scope

This package adds two capabilities to the scientific manuscript system:

1. A GitHub Actions pipeline that validates source code, executes offline tests, audits dependencies, builds API and sandbox images, generates SBOM/provenance attestations, scans images, signs publishable images with keyless Sigstore identity, and creates a protected production deployment artifact.
2. Configuration-driven domain and journal behavior for DFT/materials and AI/ML projects, with examples for Nature and Physical Review Letters.

The journal profiles are **configuration examples**, not legal or editorial guarantees. Journal requirements change, and the final submission must be checked against the current official author instructions.[1] [2] [3]

## 2. CI/CD workflow

The workflow is stored at `.github/workflows/ci-cd.yml`. Its stages are:

| Stage | Purpose |
|---|---|
| Validate | Install locked dependencies, compile Python, run offline tests, inspect Kubernetes YAML, and scan for committed secrets |
| Dependency audit | Run package vulnerability auditing and Bandit static analysis |
| Build | Build the API image with BuildKit, provenance, and SBOM generation |
| Scan | Run Trivy against pushed images and fail on high or critical unfixed findings |
| Sign | Sign published images through Cosign keyless OIDC identity |
| Sandbox image | Build, scan, attest, and sign the separate scientific runner image |
| Deploy gate | Require the protected `production` environment and verify the image signature before producing a deployment manifest |

The workflow uses least-privilege job permissions. Pull requests do not receive registry publishing behavior. Image publishing is restricted to pushes, and production is restricted to version tags plus a protected GitHub Environment.

## 3. Repository assumptions

The workflow assumes the following paths:

```text
.
├── .github/workflows/ci-cd.yml
├── deploy/Dockerfile
├── deploy/requirements.lock
├── deploy/k8s/manuscript-system.yaml
├── sandbox-runner/Dockerfile
├── src/
├── tests/
└── config/
```

If the project uses a different package layout, update the Dockerfile and workflow paths rather than weakening the validation stage.

## 4. Required GitHub settings

Create a protected `production` Environment. Require one or more reviewers, restrict deployment branches to version tags, and store only short-lived or externally managed credentials. Prefer GitHub OIDC to a cloud provider or registry rather than a long-lived deployment key.

For GHCR publishing, the workflow uses the automatic `GITHUB_TOKEN` with `packages: write`. Configure the package visibility and repository access separately. Do not grant `contents: write` unless a specific release job requires it.

For an external registry, replace the login step with a credential stored in a protected environment secret or use registry federation. Do not print the secret or pass it into build arguments.

## 5. Image security model

The workflow builds two distinct images. The API image should contain the LangGraph control plane and should not contain compilers, scientific credentials, arbitrary project folders, or host integration sockets. The sandbox image should contain only approved scientific profiles and parsers.

The workflow produces provenance and SBOM metadata, but those artifacts are useful only when the deployment system verifies them. Configure admission control to require:

| Policy | Example requirement |
|---|---|
| Signature | Image must be signed by the expected GitHub OIDC identity |
| Source | Image source repository and workflow must match the approved project |
| Digest | Deploy immutable image digest, not a mutable tag |
| Vulnerabilities | Block high/critical findings according to organizational policy |
| Provenance | Require a valid build attestation |
| Runtime | Require non-root, dropped capabilities, and restricted pod security |

## 6. CI test policy

Tests that run in pull requests must not call live literature providers, quantum hardware, DFT software, cloud services, or production LLM endpoints. Use fake provider and fake model implementations, as in the DFT end-to-end test package. Live-provider tests should be a separate manually approved or scheduled workflow with restricted credentials and explicit cost limits.

The test suite should include schema validation, provenance checks, sandbox policy tests, graph interrupt/resume tests, figure generation, literature deduplication, draft evidence-link checks, reviewer triage, and revision-loop stopping conditions.

## 7. Prompt configuration architecture

Prompt templates are stored in `config/prompts/domain_prompts.yaml`. They are separated into common rules and domain overlays. The common rules prohibit fabricated scientific content, while the domain overlay adds terminology, domain checks, writing guidance, and reviewer concerns.

A prompt request should be assembled in this order:

```text
common system rules
  + domain overlay
  + journal overlay
  + task-specific instructions
  + structured input packet
  + output schema
```

The model should receive IDs and structured records rather than unrestricted access to the entire project directory. Prompt versions must be recorded with every generated manuscript block, assessment, review, caption, and revision.

## 8. Journal configuration architecture

Journal profiles are stored in `config/journals/journals.yaml`. The profile should control section planning, writing tone, title and abstract guidance, manuscript-specific validation, citation style selection, and final human approval gates.

A journal profile must not silently alter the scientific claims. It may change organization and presentation, but claim scope, evidence links, numerical values, and provenance remain controlled by the project evidence model.

## 9. Nature example

The Nature example uses a numbered reference style with article titles required, based on the official Nature formatting guidance.[1] The profile emphasizes a concise, broad-significance summary, clear distinction between observation and interpretation, accessible terminology, and data/code availability statements.

The profile should be treated as a preparation aid. It should not encode an assumed acceptance threshold or promise a particular word limit unless the current article-type guidance has been verified.

## 10. Physical Review Letters example

The Physical Review Letters example uses a compact Letter-oriented structure, numbered bracket citations, and APS-style physical-science terminology. It emphasizes a self-contained title, a focused abstract, a compact central result, and explicit handling of supplemental material or end matter where relevant.[2] [3]

The profile deliberately does not hard-code a universal page or word limit. Such limits and submission rules should be validated against the current PRL author page and article-type requirements before release.

## 11. Citation rendering

Citation styles are stored in `config/citations/citation_styles.yaml`. The renderer should operate on verified literature records, not raw prose. Each citation should maintain a stable `literature_id`, DOI when available, source URL, metadata verification state, and access date when the source is web-only.

The renderer should support:

| Operation | Requirement |
|---|---|
| In-text citation | Convert literature IDs to the selected style’s number format |
| Bibliography ordering | Use first appearance for numbered styles |
| Metadata formatting | Apply author, title, journal, year, volume, pages, DOI rules |
| Duplicate detection | Merge DOI/title duplicates before numbering |
| Missing metadata | Flag rather than fabricate fields |
| Style validation | Confirm every cited ID has a bibliography record |
| Journal switch | Re-render from the same literature registry without rewriting scientific prose |

The examples in the citation file are placeholders and must not be used as real references.

## 12. Example configuration selection

```python
from pathlib import Path
import yaml

config_dir = Path("config")
journal_id = "nature"
domain_id = "dft_materials"

journals = yaml.safe_load((config_dir / "journals/journals.yaml").read_text())
prompts = yaml.safe_load((config_dir / "prompts/domain_prompts.yaml").read_text())
citations = yaml.safe_load((config_dir / "citations/citation_styles.yaml").read_text())

journal = journals["journals"][journal_id]
domain = prompts["domains"][domain_id]
citation = citations["styles"][journal["citation"]["style_id"]]
```

In production, use Pydantic schemas for these files and reject unknown fields. Store a hash of the selected journal profile, prompt profile, and citation style in every run record.

## 13. Change management

Prompt and journal configuration changes are scientific-output changes. Review them like code:

1. Open a pull request.
2. Run configuration schema validation.
3. Run golden manuscript fixtures.
4. Compare claim coverage, citation rendering, section structure, and reviewer findings.
5. Obtain domain-owner approval.
6. Record the configuration version in the release manifest.

Never update a prompt in production without being able to identify which manuscript blocks were generated under the previous version.

## 14. References

[1]: https://www.nature.com/nature/for-authors/formatting-guide "Nature Formatting Guide"
[2]: https://journals.aps.org/prl/authors "Physical Review Letters — Information for Authors"
[3]: https://journals.aps.org/authors/style-basics "APS Journals — Style Basics"
