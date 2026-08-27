# Manuscript System

Local-first, human-supervised LangGraph pipeline that turns a completed
research project folder into a claim-evidence readiness assessment, and later
a human-approved manuscript release package. See `docs/` for the full design
(start with `docs/Scientific_Manuscript_System_Architecture_and_Plan.md` and
`docs/LangGraph Implementation Specification.md`).

**Status:** the assessment graph runs end-to-end (intake → manifest →
discovery → domain classification → evidence extraction → audit → readiness →
human gate → route), fully checkpointed and resumable. Discovery is a
domain-agnostic structure summary (`ProjectKnowledgeMap`: file counts by role,
whether a README/environment file is present) built entirely from the
manifest's `SourceAsset` rows — it runs before domain classification, so it
can't use domain checklists (that's evidence extraction's job); absence-only
findings (`missing_readme`, `missing_environment_file`, `no_code_files_found`,
`no_data_or_figures_found`) print right at intake time now, not just buried in
a later completion plan. `run_audits` also runs a deterministic data-sufficiency
check (`tools/data_sufficiency.py`) against every scanned `.csv` file — readable,
non-empty, all-finite (`data_file_unreadable`/`data_file_empty`/
`data_file_non_finite_values`). Scoped to `.csv` only: `.dat` files in this
domain are a mixed format (some are numeric matrices, some are `key = value`
metadata pairs, confirmed against real data), so a generic tabular parser can't
validate both without false positives. (The ported `LiteratureState`/
`FigureState` subgraphs' other checks — LLM-based literature grounding/novelty,
and DFT/AI-ML metadata-completeness checks — aren't wired in: the former
duplicates the already-built citation-verification search client and risks the
LLM token budget, the latter would fail every real project by construction
since nothing here extracts that per-file metadata yet.) Evidence extraction is
deterministic (no LLM, keyword-matched against `configs/domains/*.yaml`);
`run_audits` then runs the real publishability-assessment logic from
`graphs/subgraphs/novelty_and_publishability.py` (real LLM calls) whenever
evidence was actually extracted, and stays LLM-free otherwise. Both routes out
of the human gate produce real, persisted output: `APPROVE_COMPLETION_PLAN`
turns missing-checklist-item `Finding`s and per-assessor `missing_evidence`
into a deduplicated task list (`lexical_similarity`-based dedup across
assessors; `REQUIRED` for `CENTRAL` claims, `RECOMMENDED` otherwise), and
`APPROVE_MANUSCRIPT_PLANNING` builds a section outline from the target
journal's real `required_sections` (`--journal`, falls back to plain IMRaD
without one) with claims allocated by `claim_type`. Manuscript drafting and
5-persona peer-review simulation are both wired in too — see "Draft
manuscript sections" and "Simulate peer review" below.

There's a second, separate entry point for the opposite case — a manuscript
that's already fully written but not yet submitted: `manuscript_evaluation.py`
checks citation integrity and journal-structure compliance, and — when a
linked raw-data project is given — cross-checks the manuscript's own numeric
claims against real values extracted from that project's `.csv`/`.dat`/`.ipynb`
files. See "Evaluate an already-written manuscript" below.

## Setup

```bash
conda env create -f environment.yml
conda activate manuscript-system

cp .env.example .env
# edit .env and add OPENAI_API_KEY (not required for Phase 0 tests, but
# settings.py expects the file to exist)
```

## Run the tests

```bash
PYTHONPATH=src pytest -q tests/graph/test_assessment_skeleton.py
```

## Try it against a real project (read-only)

```bash
cd /home/user/Magent-manuscript
PYTHONPATH=src python -m manuscript_system.cli intake \
  --path /home/user/Hossain/AQT_electrolyte \
  --domain quantum_chemistry \
  --journal advanced_quantum_technologies
```

`--journal` is optional — it's a target journal profile id from
`configs/journals.yaml`, used only to shape a later manuscript plan's section
outline; the assessment itself doesn't need one. This prints a `thread_id`
and pauses at the assessment-review human gate. Resume it with either
decision:

```bash
# turns missing evidence into an actionable, deduplicated task list
PYTHONPATH=src python -m manuscript_system.cli approve \
  --thread-id <thread_id from above> \
  --decision APPROVE_COMPLETION_PLAN

# builds a section outline (from --journal's required_sections, or plain
# IMRaD without one) with claims allocated by claim_type
PYTHONPATH=src python -m manuscript_system.cli approve \
  --thread-id <thread_id from above> \
  --decision APPROVE_MANUSCRIPT_PLANNING
```

Verify nothing in the source project was touched:

```bash
cd /home/user/Hossain/AQT_electrolyte && git status
```

## Draft manuscript sections

After a run has been approved for `APPROVE_MANUSCRIPT_PLANNING`, draft real
section text (one real LLM call per section) from the approved
`ManuscriptPlan`'s claim allocations:

```bash
PYTHONPATH=src python -m manuscript_system.cli draft --project-id <project_id>
```

Only sections with at least one allocated claim are drafted — an empty
section would have nothing to write from, and `WRITER_SYSTEM`'s prompt
instructs the model to abstain rather than invent content, which would block
the whole run before reaching a section that actually has claims. Skipped
sections print explicitly rather than silently vanishing. Prints a
`thread_id` and pauses at the draft-review human gate; resume with:

```bash
PYTHONPATH=src python -m manuscript_system.cli approve-draft \
  --thread-id <thread_id from above> \
  --decision APPROVE_DRAFT_FOR_PEER_REVIEW
```

## Simulate peer review

Once drafted blocks exist, run the 5-persona peer-review/revision cycle
(`graphs/subgraphs/drafting_and_peer_review.py`'s `ReviewState`) against
them:

```bash
PYTHONPATH=src python -m manuscript_system.cli review --project-id <project_id>
```

Real LLM calls: 5 (one per reviewer persona: scientific expert, critical
reviewer, statistical/computational, journal reviewer, hostile reviewer),
plus one more per accepted, block-referencing comment during revision.
`triage`/`verify`/`response-to-reviewers` are deterministic despite taking an
`llm` argument — confirmed by reading the actual node bodies before wiring
this in, since the naive "15-20+ calls" estimate assumed all of them called
the model for real. The graph pauses at up to 3 human gates in sequence;
resume each with:

```bash
PYTHONPATH=src python -m manuscript_system.cli approve-review \
  --thread-id <thread_id from above> \
  --decision <APPROVE_REVISION_PLAN|...>
```

Runs against the ported module's own in-memory `ManuscriptRepository` rather
than a real-SQLite adapter for its 6 internal record types — several of its
nodes mutate that repo via direct dict access (`repo.comments[x] = ...`), a
much tighter coupling than drafting's single `put_block()` call. LangGraph's
checkpointer already durably persists the full state (including all 6
types) across every human gate, the same mechanism trusted everywhere else
in this system; a lightweight `PeerReviewRound` summary record is persisted
separately once a round reaches `SUCCEEDED`/`BLOCKED`, for cross-run
visibility.

**Two real bugs found and fixed while verifying this against live output**
(not caught by the module's original, LLM-free tests, since both only
manifest with real, independently-generated multi-persona LLM output):
Groq's strict-schema mode was rejecting `ReviewerReport` responses because
verbose comments hit the completion-token cap mid-generation (fixed by
tightening the reviewer prompt to at most 3 short comments, not by raising
the token cap — that cap exists specifically to avoid a different, previously
-fixed rate-limit failure); and different reviewer personas independently
default to the same short comment IDs (`COMMENT-1`, `COMMENT-2`, ...), which
silently overwrote each other in the flat `comments` dict until each
persona's comment IDs were namespaced.

## Evaluate an already-written manuscript

For a complete manuscript that hasn't been submitted yet (LaTeX + bibliography
+ figures), as opposed to raw research data — checks citation integrity
(every `\cite{}` key resolves to a real bibliography entry, and every
bibliography entry is a real, findable work per OpenAlex/Crossref/arXiv, with
software/dataset citations like Gaussian or Qiskit correctly excluded rather
than reported as "unverified"), journal-structure compliance
(`configs/journals.yaml`), and citation-*formatting* compliance against the
target journal's style (`configs/citation_styles.yaml`, via the journal
profile's `citation.style_id`) — specifically, whether a bibliography entry
has a title when the style requires one, and whether a DOI this system
already found via literature search made it into the `.bib` entry. (Author
name format and et-al truncation describe the *rendered* bibliography, a
`.bst`-compile-time artifact the `.bib` source alone can't confirm, so those
aren't checked.) Deliberately LLM-free — citation verification and numeric
matching are both fuzzy/precision-matching problems, not
language-understanding ones.

```bash
PYTHONPATH=src python -m manuscript_system.cli evaluate-manuscript \
  --path /home/user/Documents/wiley \
  --journal advanced_quantum_technologies
```

**Optional — cross-check numeric claims against the real underlying data**,
with `--data-path`: every distinctive number (≥4 significant digits) stated
in the manuscript gets checked against real numeric content extracted from
the linked project's `.csv`/`.dat`/`.ipynb` files (notebook *output* cells,
not just structured data files — confirmed necessary: a real reported energy
in `wiley/` lives only in a notebook's printed output, not in any `.csv`).
Matching is precision-aware (the data must agree with the manuscript out to
the last digit it reported), not a percentage tolerance — a naive relative
tolerance was tried first and produced hundreds of false-positive matches on
this real data, since related energies cluster tightly in magnitude.

```bash
PYTHONPATH=src python -m manuscript_system.cli evaluate-manuscript \
  --path /home/user/Documents/wiley \
  --journal advanced_quantum_technologies \
  --data-path /home/user/Hossain/AQT_electrolyte
```

Resume the same way as `intake`/`approve`, with `approve-manuscript` and
choices `APPROVE_RELEASE_READY` / `APPROVE_NEEDS_REVISION` / `BLOCK_RUN`.

## Visualize a run (LangGraph Studio)

```bash
langgraph dev --no-browser
```

Then open the Studio UI link it prints
(`https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024`). The
page itself loads from LangChain's site, but it only talks to your local
server at that `baseUrl` — no LangSmith account or API key is needed, and no
run data leaves the machine unless you explicitly set `LANGSMITH_API_KEY`.
You can watch the graph execute node by node, inspect state at each step, and
see the `human_assessment_review` interrupt live. `langgraph.json` /
`src/manuscript_system/graphs/studio_entry.py` wire this up — that entry
point compiles the graph *without* our own SQLite checkpointer, since
`langgraph dev` manages persistence itself and rejects a graph that already
has one attached; `cli.py` and the tests still use our real checkpointer.

## Layout

```text
configs/            app/policy/domain-profile YAML
docs/                the design docs, diagrams, and research notes
deploy/              Dockerfile / docker-compose / k8s manifest — reference only, not wired up yet
langgraph.json       LangGraph Studio graph registration (see "Visualize a run" above)
src/manuscript_system/
  domain/            Pydantic records (Project, Claim, Evidence, Citation, Finding, ...)
  persistence/        SQLite database, repository, artifact store, LangGraph checkpointer
  graphs/
    assessment.py            raw-data intake through the human gate
    manuscript_evaluation.py  already-written-manuscript evaluation (citations + journal compliance)
    manuscript_drafting.py    wires write_next_section_node against an approved ManuscriptPlan
    manuscript_review.py      wires the 5-persona ReviewState peer-review/revision cycle
    evidence_extraction.py    deterministic domain classification + claim/evidence extraction
    studio_entry.py           LangGraph Studio entrypoint (no custom checkpointer)
    subgraphs/                novelty/publishability, drafting, and peer-review all wired in;
                               literature_and_figures.py's LLM grounding/novelty and
                               metadata-based data-sufficiency checks are not (see README body)
  tools/               filesystem scanning, LaTeX/BibTeX parsing, numeric-value extraction
                       (data_values.py), the shared OpenAI-compatible client
  cli.py               `python -m manuscript_system.cli ...`
tests/
  graph/               assessment-graph, evidence-extraction, and manuscript-evaluation tests
                        (LLM-free except one live-Groq test and the real-network citation checks)
  integration/         offline end-to-end test using the specialized subgraphs
```
