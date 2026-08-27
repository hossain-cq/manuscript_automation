from __future__ import annotations

"""Tests for tools/latex.py and manuscript_evaluation.py.

Parser unit tests are fully synthetic (no external files, no network).
The graph-level tests make real (free, keyless) OpenAlex/Crossref calls for
citation verification, same as the rest of this build's "verify for real,
don't mock the interesting part" approach - a deliberately fabricated title
should legitimately come back UNVERIFIED, which is itself a real assertion.
The real-wiley-folder test is skipped if that path doesn't exist on this
machine.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import pytest  # noqa: E402

from manuscript_system.domain.models import Citation  # noqa: E402
from manuscript_system.graphs.manuscript_evaluation import (  # noqa: E402
    build_manuscript_evaluation_graph,
    check_citation_style_compliance,
    cross_check_numeric_claims,
    initial_state,
    is_non_literature_citation,
)
from manuscript_system.persistence.database import connect  # noqa: E402
from manuscript_system.persistence.repositories import Repository, new_id  # noqa: E402
from manuscript_system.tools.data_values import (  # noqa: E402
    ExtractedValue,
    extract_numeric_values,
    find_distinctive_numbers,
    significant_digits,
)
from manuscript_system.tools.latex import (  # noqa: E402
    BibEntry,
    find_bibliography,
    find_manuscript_tex,
    parse_bibliography,
    parse_manuscript_sections,
)

WILEY_PATH = "/home/user/Documents/wiley"
AQT_ELECTROLYTE_PATH = "/home/user/Hossain/AQT_electrolyte"

SYNTHETIC_TEX = r"""
\documentclass{article}
\begin{document}
\title{A Test Manuscript}
\begin{abstract}
This is the abstract.
\end{abstract}

\section{Introduction}
This cites a real-ish thing~\cite{smith2020example} and a broken one \cite{nobody2099fake}.

%\section{Commented Out Section}
This text is still in the Introduction section, not a new one.

\section{Methodology}
We used a method \citep{smith2020example}.

\section{Conclusion}
Nothing more to say.

\bibliography{refs}
\end{document}
"""

SYNTHETIC_BIB = r"""
@article{smith2020example,
  title={Deep learning for protein structure prediction},
  author={Smith, John and Doe, Jane},
  journal={Nature},
  year={2020},
  doi={10.1000/example.doi}
}
"""


def make_repo(tmp_path: Path) -> Repository:
    return Repository(connect(str(tmp_path / "test.sqlite")))


def write_synthetic_manuscript(tmp_path: Path) -> Path:
    (tmp_path / "main.tex").write_text(SYNTHETIC_TEX)
    (tmp_path / "ref.bib").write_text(SYNTHETIC_BIB)
    return tmp_path


# --- parser unit tests (synthetic, no network) ------------------------------


def test_parse_manuscript_sections_skips_commented_out_section(tmp_path):
    tex_path = tmp_path / "main.tex"
    tex_path.write_text(SYNTHETIC_TEX)
    sections = parse_manuscript_sections(tex_path)
    names = [s.name for s in sections]
    assert names == ["Introduction", "Methodology", "Conclusion"]


def test_parse_manuscript_sections_extracts_cite_keys_per_section(tmp_path):
    tex_path = tmp_path / "main.tex"
    tex_path.write_text(SYNTHETIC_TEX)
    sections = parse_manuscript_sections(tex_path)
    intro = next(s for s in sections if s.name == "Introduction")
    assert set(intro.cite_keys) == {"smith2020example", "nobody2099fake"}
    methodology = next(s for s in sections if s.name == "Methodology")
    assert methodology.cite_keys == ["smith2020example"]


def test_parse_bibliography_reads_real_fields(tmp_path):
    bib_path = tmp_path / "ref.bib"
    bib_path.write_text(SYNTHETIC_BIB)
    entries = parse_bibliography(bib_path)
    assert "smith2020example" in entries
    entry = entries["smith2020example"]
    assert entry.title == "Deep learning for protein structure prediction"
    assert entry.year == 2020
    assert entry.doi == "10.1000/example.doi"


def test_find_manuscript_tex_prefers_revised_main(tmp_path):
    (tmp_path / "main.tex").write_text("\\documentclass{article}")
    (tmp_path / "revised_main.tex").write_text("\\documentclass{article}")
    assert find_manuscript_tex(tmp_path).name == "revised_main.tex"


def test_find_bibliography_finds_bib_file(tmp_path):
    (tmp_path / "ref.bib").write_text(SYNTHETIC_BIB)
    assert find_bibliography(tmp_path).name == "ref.bib"


# --- graph-level tests (real network for citation grounding, no LLM) -------


def test_broken_citation_key_is_flagged_without_network(tmp_path):
    """A key with no bib entry never reaches the network - this path must be
    detectable even offline."""
    write_synthetic_manuscript(tmp_path)
    repo = make_repo(tmp_path)
    graph = build_manuscript_evaluation_graph(repo, str(tmp_path / "checkpoints.sqlite"))
    thread_id = "thread-manuscript-broken"
    config = {"configurable": {"thread_id": thread_id}}
    state = initial_state(
        manuscript_path=str(tmp_path), project_id="PROJECT-TEST", run_id="RUN-TEST",
        thread_id=thread_id, journal_id=None,
    )
    result = graph.invoke(state, config=config)
    assert result["__interrupt__"]
    assert result["broken_key_count"] == 1

    citations = repo.get_citations("PROJECT-TEST")
    broken = [c for c in citations if c.verification_status == "BROKEN_KEY"]
    assert broken and broken[0].cite_key == "nobody2099fake"


def test_evaluation_reports_journal_compliance(tmp_path):
    write_synthetic_manuscript(tmp_path)
    repo = make_repo(tmp_path)
    graph = build_manuscript_evaluation_graph(repo, str(tmp_path / "checkpoints.sqlite"))
    thread_id = "thread-manuscript-compliance"
    config = {"configurable": {"thread_id": thread_id}}
    state = initial_state(
        manuscript_path=str(tmp_path), project_id="PROJECT-TEST-2", run_id="RUN-TEST-2",
        thread_id=thread_id, journal_id="advanced_quantum_technologies",
    )
    result = graph.invoke(state, config=config)
    assert result["__interrupt__"]
    # The synthetic manuscript has no "Results and discussion" or
    # "Acknowledgements"/"Conflicts of interest" sections - must be flagged.
    assert "results_and_discussion" in result["missing_sections"]
    assert "acknowledgements" in result["missing_sections"]
    assert "introduction" not in result["missing_sections"]
    assert "Citations:" in result["summary"]


@pytest.mark.skipif(not Path(WILEY_PATH).exists(), reason=f"{WILEY_PATH} not present on this machine")
def test_real_wiley_manuscript_evaluates_cleanly(tmp_path):
    repo = make_repo(tmp_path)
    graph = build_manuscript_evaluation_graph(repo, str(tmp_path / "checkpoints.sqlite"))
    thread_id = "thread-manuscript-wiley"
    config = {"configurable": {"thread_id": thread_id}}
    state = initial_state(
        manuscript_path=WILEY_PATH, project_id="PROJECT-WILEY", run_id="RUN-WILEY",
        thread_id=thread_id, journal_id="advanced_quantum_technologies",
    )
    result = graph.invoke(state, config=config)
    assert result["__interrupt__"]
    assert not result.get("error")
    # Confirmed by hand earlier: every \cite{} key in revised_main.tex has a
    # matching ref.bib entry - zero broken keys expected.
    assert result["broken_key_count"] == 0
    assert result["missing_sections"] == []
    citations = repo.get_citations("PROJECT-WILEY")
    assert len(citations) >= 70


# --- non-literature citation detection --------------------------------------


def _bib_entry(**overrides: object) -> BibEntry:
    defaults = dict(
        cite_key="key1", title="Some Title", authors="A. Author", year=2020,
        doi=None, journal=None, entry_type="article", booktitle=None, note=None,
    )
    defaults.update(overrides)
    return BibEntry(**defaults)  # type: ignore[arg-type]


def test_misc_entry_with_no_journal_is_non_literature():
    entry = _bib_entry(
        entry_type="misc", journal=None, title="Gaussian~16 Revision C.01",
        note="Gaussian Inc. Wallingford CT",
    )
    flagged, reason = is_non_literature_citation(entry)
    assert flagged
    assert "software" in reason.lower() or "not identifiable" in reason.lower()


def test_misc_entry_with_journal_is_not_misclassified():
    """Grounded in a real case in wiley/ref.bib: cava2021introduction is
    @misc but has a real journal field (Chemical Reviews) - a bib-authoring
    quirk, not a software citation. Must not be flagged."""
    entry = _bib_entry(entry_type="misc", journal="Chemical Reviews", title="Introduction: quantum materials")
    flagged, _ = is_non_literature_citation(entry)
    assert not flagged


def test_article_entry_is_never_non_literature():
    entry = _bib_entry(entry_type="article", journal="Nature")
    flagged, _ = is_non_literature_citation(entry)
    assert not flagged


def _citation(**overrides: object) -> Citation:
    defaults = dict(
        citation_id=new_id("CITATION"), project_id="PROJECT-TEST", cite_key="key1",
        bib_title="A Real Title", bib_authors="Author, A.", bib_year=2020, bib_doi=None,
        verification_status="VERIFIED", verified_title="A Real Title", verified_doi=None, match_confidence=0.9,
    )
    defaults.update(overrides)
    return Citation(**defaults)  # type: ignore[arg-type]


def test_citation_style_flags_missing_title_when_style_requires_it():
    style = {"validation": {"require_title": True, "require_doi_when_available": False}}
    citation = _citation(bib_title="")
    findings = check_citation_style_compliance([citation], style, "nature-numbered", "PROJECT-TEST")
    assert len(findings) == 1
    assert findings[0].rule_id == "citation_style_missing_title"
    assert findings[0].severity == "LOW"


def test_citation_style_does_not_flag_missing_title_when_style_does_not_require_it():
    style = {"validation": {"require_title": False, "require_doi_when_available": False}}
    citation = _citation(bib_title="")
    findings = check_citation_style_compliance([citation], style, "aps-prl", "PROJECT-TEST")
    assert findings == []


def test_citation_style_flags_known_doi_missing_from_bib_entry():
    """Grounded in the real wiley/ref.bib: only 1 of 77 entries has a `doi`
    field, while aps-prl (the AQT journal's style) requires one whenever the
    literature search actually found one (verified_doi set)."""
    style = {"validation": {"require_title": False, "require_doi_when_available": True}}
    citation = _citation(bib_doi=None, verified_doi="10.1000/example")
    findings = check_citation_style_compliance([citation], style, "aps-prl", "PROJECT-TEST")
    assert len(findings) == 1
    assert findings[0].rule_id == "citation_style_missing_doi"


def test_citation_style_does_not_flag_doi_already_present():
    style = {"validation": {"require_title": False, "require_doi_when_available": True}}
    citation = _citation(bib_doi="10.1000/example", verified_doi="10.1000/example")
    findings = check_citation_style_compliance([citation], style, "aps-prl", "PROJECT-TEST")
    assert findings == []


def test_citation_style_does_not_flag_doi_when_none_was_ever_verified():
    """UNVERIFIED citations never got a verified_doi - nothing to compare
    against, so no finding (absence of a DOI in the .bib isn't itself a
    violation; only a *known* missing one is)."""
    style = {"validation": {"require_title": False, "require_doi_when_available": True}}
    citation = _citation(bib_doi=None, verified_doi=None, verification_status="UNVERIFIED")
    findings = check_citation_style_compliance([citation], style, "aps-prl", "PROJECT-TEST")
    assert findings == []


def test_citation_style_skips_broken_key_citations():
    style = {"validation": {"require_title": True, "require_doi_when_available": True}}
    citation = _citation(
        bib_title="", bib_doi=None, verified_doi=None, verification_status="BROKEN_KEY",
    )
    findings = check_citation_style_compliance([citation], style, "nature-numbered", "PROJECT-TEST")
    assert findings == []


def test_citation_style_returns_nothing_without_a_style():
    citation = _citation(bib_title="")
    findings = check_citation_style_compliance([citation], None, None, "PROJECT-TEST")
    assert findings == []


def test_non_literature_citation_is_flagged_without_touching_network(tmp_path):
    tex = SYNTHETIC_TEX + "\n"  # reuse existing fixture, add a software cite
    tex_with_software = tex.replace(
        r"\section{Conclusion}", r"We used \cite{gaussian16} for calculations.\section{Conclusion}"
    )
    (tmp_path / "main.tex").write_text(tex_with_software)
    (tmp_path / "ref.bib").write_text(
        SYNTHETIC_BIB
        + '\n@misc{gaussian16,\n  title={Gaussian~16 Revision C.01},\n  note={Gaussian Inc. Wallingford CT},\n  year={2016}\n}\n'
    )
    repo = make_repo(tmp_path)
    graph = build_manuscript_evaluation_graph(repo, str(tmp_path / "checkpoints.sqlite"))
    thread_id = "thread-manuscript-nonlit"
    config = {"configurable": {"thread_id": thread_id}}
    state = initial_state(
        manuscript_path=str(tmp_path), project_id="PROJECT-NONLIT", run_id="RUN-NONLIT",
        thread_id=thread_id, journal_id=None,
    )
    result = graph.invoke(state, config=config)
    assert result["__interrupt__"]
    assert result["non_literature_count"] == 1
    citations = repo.get_citations("PROJECT-NONLIT")
    gaussian = next(c for c in citations if c.cite_key == "gaussian16")
    assert gaussian.verification_status == "NON_LITERATURE"


@pytest.mark.skipif(not Path(WILEY_PATH).exists(), reason=f"{WILEY_PATH} not present on this machine")
def test_real_wiley_arxiv_preprint_gets_verified_via_arxiv_fallback(tmp_path):
    """cao2021towards in the real wiley/ref.bib is an arXiv-only preprint
    (journal={arXiv preprint arXiv:2109.02110}) that Crossref/OpenAlex don't
    reliably resolve by title - confirmed the arXiv fallback finds it."""
    from manuscript_system.graphs.manuscript_evaluation import _verify_against_literature
    from manuscript_system.graphs.subgraphs.novelty_and_publishability import LiteratureClient
    from manuscript_system.tools.latex import find_bibliography as find_bib

    bib = parse_bibliography(find_bib(WILEY_PATH))
    entry = bib["cao2021towards"]
    client = LiteratureClient(cache_dir=str(tmp_path / "literature_cache"))
    record, score = _verify_against_literature(client, entry.title)
    assert record is not None
    assert score >= 0.5
    assert record.provider in ("ARXIV", "CROSSREF", "OPENALEX")


# --- numeric cross-checking: tools/data_values.py (synthetic, no network) --


def test_significant_digits_filters_low_precision_numbers():
    assert significant_digits("-945.0931") == 7
    assert significant_digits("1.50") == 3
    assert significant_digits("0.001") == 1


def test_find_distinctive_numbers_excludes_low_precision():
    text = "The bond length was 1.50 Angstrom and the energy was -945.093125 Hartree."
    found = find_distinctive_numbers(text)
    assert "-945.093125" in found
    assert "1.50" not in found


def test_extract_numeric_values_from_csv(tmp_path):
    (tmp_path / "results.csv").write_text("distance,energy\n1.5,-945.093125\n1.7,-945.201847\n")
    values = extract_numeric_values(tmp_path)
    assert any(v.value == -945.093125 for v in values)
    # distance column (1.5, 1.7) has too few significant digits to be distinctive
    assert not any(v.value == 1.5 for v in values)


def test_extract_numeric_values_from_dat_skips_comments(tmp_path):
    (tmp_path / "results.dat").write_text("# header comment 945.999999\n1.5000  -945.039201\n")
    values = extract_numeric_values(tmp_path)
    assert any(v.value == -945.039201 for v in values)
    assert not any(v.value == 945.999999 for v in values)  # comment line must be skipped


def test_extract_numeric_values_from_ipynb_output(tmp_path):
    import json
    notebook = {
        "cells": [{
            "cell_type": "code", "source": ["print('hi')"],
            "outputs": [{"output_type": "stream", "text": ["CASCI E = -944.740594346671\n"]}],
        }],
    }
    (tmp_path / "notebook.ipynb").write_text(json.dumps(notebook))
    values = extract_numeric_values(tmp_path)
    assert any(v.value == -944.740594346671 for v in values)


# --- numeric cross-checking: matching logic (synthetic, no network) --------


def test_cross_check_matches_at_manuscript_precision():
    pool = [ExtractedValue(value=-945.09312277, source_relative_path="notebooks/x.ipynb", context="CASCI E = -945.09312277")]
    checks = cross_check_numeric_claims([("-945.0931", "Results")], pool)
    assert len(checks) == 1
    assert checks[0].status == "SUPPORTED_BY_DATA"
    assert checks[0].matched_source_path == "notebooks/x.ipynb"


def test_cross_check_does_not_match_unrelated_close_value():
    """The bug this guards against: a naive relative-tolerance match would
    treat -945.0392 as "close enough" to -945.0931 (both ~945 in magnitude);
    precision-aware matching correctly rejects it."""
    pool = [ExtractedValue(value=-945.03920133, source_relative_path="data/x.dat", context="")]
    checks = cross_check_numeric_claims([("-945.0931", "Results")], pool)
    assert checks[0].status == "NOT_FOUND_IN_DATA"


def test_cross_check_fabricated_value_not_found():
    pool = [ExtractedValue(value=-945.09312277, source_relative_path="notebooks/x.ipynb", context="")]
    checks = cross_check_numeric_claims([("-999.1234", "Results")], pool)
    assert checks[0].status == "NOT_FOUND_IN_DATA"
    assert checks[0].matched_value is None


# --- graph-level cross-check test (synthetic, no network) ------------------


def test_graph_cross_checks_manuscript_against_linked_data(tmp_path):
    manuscript_dir = tmp_path / "manuscript"
    data_dir = tmp_path / "data"
    manuscript_dir.mkdir()
    data_dir.mkdir()

    tex = SYNTHETIC_TEX.replace(
        r"\section{Conclusion}",
        r"The computed ground-state energy was -945.093125 Ha, matching prior work.\section{Conclusion}",
    )
    (manuscript_dir / "main.tex").write_text(tex)
    (manuscript_dir / "ref.bib").write_text(SYNTHETIC_BIB)
    (data_dir / "results.csv").write_text("system,energy\nLiPF6,-945.093125\n")

    repo = make_repo(tmp_path)
    graph = build_manuscript_evaluation_graph(repo, str(tmp_path / "checkpoints.sqlite"))
    thread_id = "thread-manuscript-crosscheck"
    config = {"configurable": {"thread_id": thread_id}}
    state = initial_state(
        manuscript_path=str(manuscript_dir), project_id="PROJECT-XCHECK", run_id="RUN-XCHECK",
        thread_id=thread_id, journal_id=None, linked_data_path=str(data_dir),
    )
    result = graph.invoke(state, config=config)
    assert result["__interrupt__"]
    assert result["numeric_claims_found"] >= 1
    assert result["numeric_claims_supported"] >= 1
    assert "Numeric claims:" in result["summary"]

    checks = repo.get_numeric_cross_checks("PROJECT-XCHECK")
    supported = [c for c in checks if c.status == "SUPPORTED_BY_DATA"]
    assert any(c.claimed_text == "-945.093125" for c in supported)


def test_graph_without_linked_data_path_behaves_as_before(tmp_path):
    """No regression: omitting --data-path must produce the exact same
    manuscript-only behavior as before this feature existed."""
    write_synthetic_manuscript(tmp_path)
    repo = make_repo(tmp_path)
    graph = build_manuscript_evaluation_graph(repo, str(tmp_path / "checkpoints.sqlite"))
    thread_id = "thread-manuscript-nodatapath"
    config = {"configurable": {"thread_id": thread_id}}
    state = initial_state(
        manuscript_path=str(tmp_path), project_id="PROJECT-NODATA", run_id="RUN-NODATA",
        thread_id=thread_id, journal_id=None,
    )
    result = graph.invoke(state, config=config)
    assert result["__interrupt__"]
    assert "Numeric claims:" not in result["summary"]
    assert repo.get_numeric_cross_checks("PROJECT-NODATA") == []


@pytest.mark.skipif(
    not (Path(WILEY_PATH).exists() and Path(AQT_ELECTROLYTE_PATH).exists()),
    reason="requires both the real wiley/ manuscript and AQT_electrolyte data on this machine",
)
def test_real_manuscript_data_cross_check(tmp_path):
    repo = make_repo(tmp_path)
    graph = build_manuscript_evaluation_graph(repo, str(tmp_path / "checkpoints.sqlite"))
    thread_id = "thread-manuscript-real-crosscheck"
    config = {"configurable": {"thread_id": thread_id}}
    state = initial_state(
        manuscript_path=WILEY_PATH, project_id="PROJECT-REAL-XCHECK", run_id="RUN-REAL-XCHECK",
        thread_id=thread_id, journal_id="advanced_quantum_technologies", linked_data_path=AQT_ELECTROLYTE_PATH,
    )
    result = graph.invoke(state, config=config)
    assert result["__interrupt__"]
    assert not result.get("error")
    # Confirmed by hand: "-945.0931" (a real reported CASCI energy) matches
    # real notebook output in AQT_electrolyte - at least some real numeric
    # claims must be found supported, not just found.
    assert result["numeric_claims_found"] > 0
    assert result["numeric_claims_supported"] > 0
