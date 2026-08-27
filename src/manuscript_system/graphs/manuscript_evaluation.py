from __future__ import annotations

import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from ..domain.models import Citation, Finding, NumericCrossCheck
from ..persistence.checkpointer import build_checkpointer
from ..persistence.repositories import Repository, new_id
from ..tools.data_values import extract_numeric_values, find_distinctive_numbers
from ..tools.citation_styles import load_citation_style
from ..tools.filesystem import BoundaryError, validate_project_path
from ..tools.journals import load_journal_profile
from ..tools.latex import find_bibliography, find_manuscript_tex, parse_bibliography, parse_manuscript_sections
import requests

from .subgraphs.novelty_and_publishability import LiteratureClient, VerifiedLiteratureRecord, lexical_similarity

"""Manuscript evaluation graph: citation integrity, journal-structure
compliance, and (when a linked raw-data project is given) numeric
cross-checking, for an already-written manuscript - as opposed to
assessment.py (which starts from raw research data and works toward a
manuscript).

Citation/structure checks are deliberately LLM-free (like Phase 0's
zero-LLM-dependency assessment path) - fuzzy-matching against real
literature records is not a language-understanding problem. Numeric
cross-checking is the same: matching a manuscript-stated number against real
data-file content is a precision-matching problem, not a semantic one -
see cross_check_numeric_claims's docstring for how "match" is defined and
why a naive relative-tolerance approach doesn't work.
"""

MANUSCRIPT_CHOICES = (
    "APPROVE_RELEASE_READY",
    "APPROVE_NEEDS_REVISION",
    "BLOCK_RUN",
)


class RunContext(TypedDict, total=False):
    project_id: str
    run_id: str
    thread_id: str
    workflow_name: str
    status: str
    current_stage: str
    created_at: str
    updated_at: str


class ManuscriptEvaluationState(TypedDict, total=False):
    context: RunContext
    manuscript_path: str
    journal_id: str | None
    linked_data_path: str | None
    tex_path: str | None
    bib_path: str | None
    citation_ids: list[str]
    verified_count: int
    unverified_count: int
    broken_key_count: int
    non_literature_count: int
    missing_sections: list[str]
    citation_style_id: str | None
    citation_style_finding_count: int
    numeric_check_ids: list[str]
    numeric_claims_found: int
    numeric_claims_supported: int
    finding_ids: list[str]
    summary: str | None
    human_decision: dict[str, Any] | None
    error: dict[str, Any] | None


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_section_name(name: str) -> str:
    return re.sub(r"[^a-z]+", " ", name.lower()).strip()


def _best_match(records: list[VerifiedLiteratureRecord], title: str) -> tuple[VerifiedLiteratureRecord | None, float]:
    best, best_score = None, 0.0
    for record in records:
        score = lexical_similarity(title, record.title)
        if score > best_score:
            best, best_score = record, score
    return best, best_score


def _search_and_match(
    title: str, best_so_far: float,
    fetch: Any, normalize: Any,
) -> tuple[VerifiedLiteratureRecord | None, float]:
    try:
        items = fetch()
    except requests.RequestException:
        items = []
    records = [r for item in items if (r := normalize(item, title))]
    return _best_match(records, title)


def _verify_against_literature(
    client: LiteratureClient, title: str, match_threshold: float = 0.5,
) -> tuple[VerifiedLiteratureRecord | None, float]:
    """Crossref first, then OpenAlex, then arXiv - each only as a fallback
    when nothing so far already gives a confident match.

    search_verified() (used elsewhere in this codebase for single-claim
    novelty comparison) always queries every provider - fine for a handful
    of calls, but a real bibliography can have 50-80+ entries, and firing
    every provider for every one of them triggers OpenAlex's anonymous rate
    limit hard (confirmed: a real run took 38 minutes with both providers
    always queried, most of it spent in retry backoff). Crossref alone
    resolves the large majority of real citations; arXiv is checked last
    since it only helps preprints, which are the minority even in a
    quantum-computing bibliography.
    """
    if not title:
        return None, 0.0

    best, best_score = _search_and_match(
        title, 0.0, lambda: client.crossref_search(title, rows=5), client.normalize_crossref
    )
    if best_score >= match_threshold:
        return best, best_score

    openalex_best, openalex_score = _search_and_match(
        title, best_score, lambda: client.openalex_search(title, per_page=5), client.normalize_openalex
    )
    if openalex_score > best_score:
        best, best_score = openalex_best, openalex_score
    if best_score >= match_threshold:
        return best, best_score

    arxiv_best, arxiv_score = _search_and_match(
        title, best_score, lambda: client.arxiv_search(title, max_results=3), client.normalize_arxiv
    )
    if arxiv_score > best_score:
        best, best_score = arxiv_best, arxiv_score
    return best, best_score


_SOFTWARE_TITLE_HINTS = re.compile(
    r"\b(revision|version|v\d+\.\d+|release)\b|Inc\.|Zenodo", re.IGNORECASE
)


def is_non_literature_citation(entry: Any) -> tuple[bool, str]:
    """Identify bibliography entries that aren't papers - software, code
    releases, database records - so they're never sent to a literature
    search that can only ever fail to "verify" them.

    Grounded in the real wiley/ref.bib: entries with no journal/booktitle
    are the pattern, not bare entry_type=='misc' alone - one real @misc
    entry there (cava2021introduction) is an actual Chemical Reviews
    article with a journal field, just miscategorized by whoever wrote the
    .bib file. Requiring "no journal AND no booktitle" avoids
    misclassifying it as non-literature.
    """
    if entry.journal or entry.booktitle:
        return False, ""
    if entry.entry_type.lower() in ("misc", "software", "online"):
        if entry.note or _SOFTWARE_TITLE_HINTS.search(entry.title or ""):
            return True, f"BibTeX @{entry.entry_type} entry with no journal/booktitle - looks like a software/tool/dataset citation, not a paper."
        return True, f"BibTeX @{entry.entry_type} entry with no journal/booktitle - not identifiable as a journal article."
    return False, ""


def check_citation_style_compliance(
    citations: list[Citation], style: dict[str, Any] | None, style_id: str | None, project_id: str,
) -> list[Finding]:
    """Only checks what's derivable from already-parsed data without
    compiling LaTeX: whether a bib entry has a title (when the style
    requires one) and whether a DOI this system already found via
    literature search (verified_doi) made it into the .bib entry (bib_doi)
    when the style requires a DOI whenever one is available. Author-name
    format and et-al truncation describe the *rendered* bibliography (a
    .bst-compile-time artifact), not the .bib source - not checkable here.
    """
    if not style:
        return []
    validation = style.get("validation", {})
    findings: list[Finding] = []
    for citation in citations:
        if citation.verification_status == "BROKEN_KEY":
            continue
        if validation.get("require_title") and not citation.bib_title:
            findings.append(Finding(
                finding_id=new_id("FINDING"), project_id=project_id, severity="LOW",
                rule_id="citation_style_missing_title",
                message=f"Citation '{citation.cite_key}' has no title field; style '{style_id}' requires article titles in the reference list.",
                blocking=False,
            ))
        if validation.get("require_doi_when_available") and citation.verified_doi and not citation.bib_doi:
            findings.append(Finding(
                finding_id=new_id("FINDING"), project_id=project_id, severity="LOW",
                rule_id="citation_style_missing_doi",
                message=f"Citation '{citation.cite_key}' has a known DOI ({citation.verified_doi}) not recorded in the bibliography entry; style '{style_id}' requires a DOI when available.",
                blocking=False,
            ))
    return findings


def _decimal_places(text: str) -> int:
    return len(text.split(".", 1)[1]) if "." in text else 0


def cross_check_numeric_claims(
    manuscript_numbers: list[tuple[str, str]], data_pool: list[Any],
) -> list[NumericCrossCheck]:
    """manuscript_numbers is (raw_text, section_name) pairs. Matches a
    manuscript number against the data pool by rounding both to the
    precision *the manuscript itself reported* and checking for equality -
    not a relative/percentage tolerance.

    A relative tolerance doesn't work here: tried 0.1% first, and on real
    data it produced 370 "matches" for one number, because energies in the
    same study cluster tightly in absolute magnitude (~945 Hartree) and
    0.1% of 945 is nearly 1 - far looser than the actual precision at stake.
    What "matches" really means for this check is "the data agrees with the
    manuscript out to the last digit it bothered to report" - e.g. the
    manuscript's "-945.0931" (4 decimal places) should match a raw value of
    -945.09312277 once *that* is rounded to 4 places, not any value within
    an arbitrary percentage. Confirmed against the real AQT_electrolyte
    project: this correctly matches "-945.0931" to real notebook output and
    correctly finds zero matches for a fabricated control value.
    """
    results: list[NumericCrossCheck] = []
    for raw_text, section in manuscript_numbers:
        claimed = float(raw_text)
        places = _decimal_places(raw_text)
        match = next((v for v in data_pool if round(v.value, places) == round(claimed, places)), None)
        if match:
            results.append(NumericCrossCheck(
                check_id=new_id("NUMCHECK"), project_id="", claimed_value=claimed, claimed_text=raw_text,
                source_section=section, status="SUPPORTED_BY_DATA", matched_value=match.value,
                matched_source_path=match.source_relative_path, matched_context=match.context,
                precision_places=places,
            ))
        else:
            results.append(NumericCrossCheck(
                check_id=new_id("NUMCHECK"), project_id="", claimed_value=claimed, claimed_text=raw_text,
                source_section=section, status="NOT_FOUND_IN_DATA", precision_places=places,
            ))
    return results


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


def initialize_run(state: ManuscriptEvaluationState) -> dict[str, Any]:
    return {
        "context": {**state["context"], "status": "RUNNING", "current_stage": "BOUNDARY_VALIDATION", "updated_at": now_iso()},
        "error": None,
    }


def validate_paths(state: ManuscriptEvaluationState) -> dict[str, Any]:
    try:
        root = validate_project_path(state["manuscript_path"])
    except BoundaryError as exc:
        return {
            "context": {**state["context"], "status": "BLOCKED", "updated_at": now_iso()},
            "error": {"code": "INVALID_MANUSCRIPT_PATH", "message": str(exc)},
        }
    tex_path = find_manuscript_tex(root)
    if tex_path is None:
        return {
            "context": {**state["context"], "status": "BLOCKED", "updated_at": now_iso()},
            "error": {"code": "NO_MANUSCRIPT_FOUND", "message": f"No .tex file found under {root}"},
        }
    bib_path = find_bibliography(root)
    return {
        "tex_path": str(tex_path),
        "bib_path": str(bib_path) if bib_path else None,
        "context": {**state["context"], "current_stage": "EVALUATION", "updated_at": now_iso()},
    }


def evaluate_manuscript(state: ManuscriptEvaluationState, repo: Repository) -> dict[str, Any]:
    project_id = state["context"]["project_id"]
    sections = parse_manuscript_sections(state["tex_path"])
    bib = parse_bibliography(state["bib_path"]) if state.get("bib_path") else {}
    raw_tex = Path(state["tex_path"]).read_text(encoding="utf-8")

    citations: list[Citation] = []
    findings: list[Finding] = []
    client = LiteratureClient()

    all_cite_keys = sorted({key for section in sections for key in section.cite_keys})
    for key in all_cite_keys:
        entry = bib.get(key)
        if entry is None:
            citations.append(Citation(
                citation_id=new_id("CITATION"), project_id=project_id, cite_key=key,
                verification_status="BROKEN_KEY", notes="No matching entry in the bibliography.",
            ))
            findings.append(Finding(
                finding_id=new_id("FINDING"), project_id=project_id, severity="HIGH",
                rule_id="broken_citation_key",
                message=f"\\cite{{{key}}} is used but has no matching bibliography entry.",
                blocking=False,
            ))
            continue

        non_literature, non_literature_reason = is_non_literature_citation(entry)
        if non_literature:
            citations.append(Citation(
                citation_id=new_id("CITATION"), project_id=project_id, cite_key=key,
                bib_title=entry.title, bib_authors=entry.authors, bib_year=entry.year, bib_doi=entry.doi,
                verification_status="NON_LITERATURE", notes=non_literature_reason,
            ))
            continue

        time.sleep(0.2)  # courtesy delay - see _verify_against_literature's docstring
        best_record, best_score = _verify_against_literature(client, entry.title)

        if best_record is not None and best_score >= 0.5:
            citations.append(Citation(
                citation_id=new_id("CITATION"), project_id=project_id, cite_key=key,
                bib_title=entry.title, bib_authors=entry.authors, bib_year=entry.year, bib_doi=entry.doi,
                verification_status="VERIFIED", verified_title=best_record.title, verified_doi=best_record.doi,
                match_confidence=best_score,
            ))
        else:
            citations.append(Citation(
                citation_id=new_id("CITATION"), project_id=project_id, cite_key=key,
                bib_title=entry.title, bib_authors=entry.authors, bib_year=entry.year, bib_doi=entry.doi,
                verification_status="UNVERIFIED", match_confidence=best_score,
                notes="No closely matching record found via OpenAlex/Crossref search - not evidence of fabrication.",
            ))
            findings.append(Finding(
                finding_id=new_id("FINDING"), project_id=project_id, severity="MEDIUM",
                rule_id="unverified_citation",
                message=f"Citation '{key}' ({entry.title[:80]!r}) could not be verified against literature search.",
                blocking=False,
            ))

    # Journal structural compliance - deterministic, against configs/journals.yaml.
    missing_sections: list[str] = []
    profile = load_journal_profile(state.get("journal_id"))
    if profile:
        found_names = {_normalize_section_name(s.name) for s in sections}
        has_title = bool(re.search(r"\\title\s*\{", raw_tex))
        has_abstract = bool(re.search(r"\\begin\{abstract\}", raw_tex))
        has_references = bool(re.search(r"\\bibliography\{|\\printbibliography|\\begin\{thebibliography\}", raw_tex))
        front_back_matter = {"title": has_title, "abstract": has_abstract, "references": has_references}

        for required in profile.get("manuscript_structure", {}).get("required_sections", []):
            if required in front_back_matter:
                if not front_back_matter[required]:
                    missing_sections.append(required)
                continue
            if _normalize_section_name(required.replace("_", " ")) not in found_names:
                missing_sections.append(required)

        for missing in missing_sections:
            findings.append(Finding(
                finding_id=new_id("FINDING"), project_id=project_id, severity="HIGH",
                rule_id=f"missing_required_section:{missing}",
                message=f"Journal profile '{state.get('journal_id')}' requires a '{missing}' section; none found.",
                blocking=False,
            ))

    # Citation-formatting compliance - deterministic, against
    # configs/citation_styles.yaml (the style the journal profile points to).
    style_id = (profile or {}).get("citation", {}).get("style_id")
    style = load_citation_style(style_id)
    style_findings = check_citation_style_compliance(citations, style, style_id, project_id)
    findings.extend(style_findings)

    repo.add_citations(citations)
    for finding in findings:
        repo.add_finding(finding)

    verified = sum(1 for c in citations if c.verification_status == "VERIFIED")
    unverified = sum(1 for c in citations if c.verification_status == "UNVERIFIED")
    broken = sum(1 for c in citations if c.verification_status == "BROKEN_KEY")
    non_literature_count = sum(1 for c in citations if c.verification_status == "NON_LITERATURE")

    return {
        "citation_ids": [c.citation_id for c in citations],
        "verified_count": verified,
        "unverified_count": unverified,
        "broken_key_count": broken,
        "non_literature_count": non_literature_count,
        "missing_sections": missing_sections,
        "citation_style_id": style_id,
        "citation_style_finding_count": len(style_findings),
        "finding_ids": [f.finding_id for f in findings],
        "context": {**state["context"], "current_stage": "NUMERIC_CROSS_CHECK", "updated_at": now_iso()},
    }


def cross_check_against_data(state: ManuscriptEvaluationState, repo: Repository) -> dict[str, Any]:
    """No-op (returns nothing new) when linked_data_path isn't set - a plain
    manuscript-only evaluation behaves exactly as it did before this node
    existed."""
    linked_data_path = state.get("linked_data_path")
    if not linked_data_path:
        return {"context": {**state["context"], "current_stage": "REPORT_SYNTHESIS", "updated_at": now_iso()}}

    project_id = state["context"]["project_id"]
    try:
        data_root = validate_project_path(linked_data_path)
    except BoundaryError:
        # A bad linked-data path shouldn't fail the whole evaluation - the
        # manuscript-only results (citations, structure) are still valid.
        return {"context": {**state["context"], "current_stage": "REPORT_SYNTHESIS", "updated_at": now_iso()}}

    sections = parse_manuscript_sections(state["tex_path"])
    manuscript_numbers = [
        (token, section.name) for section in sections for token in find_distinctive_numbers(section.text)
    ]
    data_pool = extract_numeric_values(data_root)
    checks = [
        c.model_copy(update={"project_id": project_id})
        for c in cross_check_numeric_claims(manuscript_numbers, data_pool)
    ]
    repo.add_numeric_cross_checks(checks)
    supported = sum(1 for c in checks if c.status == "SUPPORTED_BY_DATA")

    return {
        "numeric_check_ids": [c.check_id for c in checks],
        "numeric_claims_found": len(checks),
        "numeric_claims_supported": supported,
        "context": {**state["context"], "current_stage": "REPORT_SYNTHESIS", "updated_at": now_iso()},
    }


def synthesize_report(state: ManuscriptEvaluationState) -> dict[str, Any]:
    total = (
        state.get("verified_count", 0) + state.get("unverified_count", 0)
        + state.get("broken_key_count", 0) + state.get("non_literature_count", 0)
    )
    lines = [
        f"Citations: {total} used - {state.get('verified_count', 0)} verified, "
        f"{state.get('unverified_count', 0)} unverified, {state.get('broken_key_count', 0)} broken key(s), "
        f"{state.get('non_literature_count', 0)} non-literature (software/data, not searched).",
    ]
    if state.get("journal_id"):
        missing = state.get("missing_sections", [])
        lines.append(
            f"Journal '{state['journal_id']}': "
            + (f"missing section(s): {', '.join(missing)}" if missing else "all required sections present")
        )
    if state.get("citation_style_id"):
        style_count = state.get("citation_style_finding_count", 0)
        lines.append(
            f"Citation style '{state['citation_style_id']}': "
            + (f"{style_count} formatting issue(s)." if style_count else "no formatting issues.")
        )
    if state.get("linked_data_path"):
        found = state.get("numeric_claims_found", 0)
        supported = state.get("numeric_claims_supported", 0)
        lines.append(
            f"Numeric claims: {found} found - {supported} matched in the linked data, "
            f"{found - supported} not found (not evidence of fabrication - many reported values are "
            f"derived/summary numbers never saved verbatim to a file)."
        )
    summary = " ".join(lines)
    return {
        "summary": summary,
        "context": {
            **state["context"], "current_stage": "HUMAN_REVIEW", "status": "WAITING_FOR_HUMAN", "updated_at": now_iso(),
        },
    }


def human_review(state: ManuscriptEvaluationState, repo: Repository) -> dict[str, Any]:
    from ..domain.models import HumanDecision

    payload = {
        "kind": "MANUSCRIPT_EVALUATION_REVIEW",
        "question": "Review the manuscript evaluation.",
        "summary": state.get("summary"),
        "broken_key_count": state.get("broken_key_count", 0),
        "unverified_count": state.get("unverified_count", 0),
        "missing_sections": state.get("missing_sections", []),
        "choices": list(MANUSCRIPT_CHOICES),
    }
    response = interrupt(payload)
    decision = response.get("decision") if isinstance(response, dict) else response
    if decision not in MANUSCRIPT_CHOICES:
        raise ValueError(f"Invalid human decision: {decision!r}")

    repo.add_human_decision(HumanDecision(
        decision_id=new_id("DECISION"), project_id=state["context"]["project_id"],
        run_id=state["context"]["run_id"], kind=payload["kind"], decision=decision, payload=payload,
    ))
    return {
        "human_decision": {"decision": decision},
        "context": {
            **state["context"], "status": "SUCCEEDED", "current_stage": f"DONE:{decision}", "updated_at": now_iso(),
        },
    }


def blocked_terminal(state: ManuscriptEvaluationState) -> dict[str, Any]:
    return {
        "context": {**state["context"], "status": "BLOCKED", "current_stage": "BLOCKED", "updated_at": now_iso()}
    }


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def build_manuscript_evaluation_graph(repo: Repository, checkpoint_db_path: str | None = None):
    """checkpoint_db_path=None compiles without a checkpointer - same
    convention as graphs/assessment.py, for LangGraph Studio compatibility."""
    builder = StateGraph(ManuscriptEvaluationState)

    builder.add_node("initialize_run", initialize_run)
    builder.add_node("validate_paths", validate_paths)
    builder.add_node("evaluate_manuscript", lambda s: evaluate_manuscript(s, repo))
    builder.add_node("cross_check_against_data", lambda s: cross_check_against_data(s, repo))
    builder.add_node("synthesize_report", synthesize_report)
    builder.add_node("human_review", lambda s: human_review(s, repo))
    builder.add_node("blocked", blocked_terminal)

    builder.add_edge(START, "initialize_run")
    builder.add_edge("initialize_run", "validate_paths")
    builder.add_conditional_edges(
        "validate_paths",
        lambda state: "continue" if not state.get("error") else "blocked",
        {"continue": "evaluate_manuscript", "blocked": "blocked"},
    )
    builder.add_edge("evaluate_manuscript", "cross_check_against_data")
    builder.add_edge("cross_check_against_data", "synthesize_report")
    builder.add_edge("synthesize_report", "human_review")
    builder.add_edge("human_review", END)
    builder.add_edge("blocked", END)

    if checkpoint_db_path is None:
        return builder.compile()
    return builder.compile(checkpointer=build_checkpointer(checkpoint_db_path))


def initial_state(
    *, manuscript_path: str, project_id: str, run_id: str, thread_id: str,
    journal_id: str | None, linked_data_path: str | None = None,
) -> ManuscriptEvaluationState:
    return {
        "context": {
            "project_id": project_id, "run_id": run_id, "thread_id": thread_id,
            "workflow_name": "manuscript_evaluation", "status": "CREATED", "current_stage": "CREATED",
            "created_at": now_iso(), "updated_at": now_iso(),
        },
        "manuscript_path": manuscript_path,
        "journal_id": journal_id,
        "linked_data_path": linked_data_path,
    }
