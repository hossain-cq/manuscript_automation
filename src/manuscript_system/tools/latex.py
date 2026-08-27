from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import bibtexparser

"""Deterministic (no-LLM) LaTeX manuscript and BibTeX parsing.

Deliberately simple regex-based parsing, not a full LaTeX AST - good enough
to recover section boundaries, readable text, and citation keys; not good
enough to render the document. bibtexparser handles the actual .bib grammar
(nested braces, varied entry types) rather than hand-rolling it - a well-known
source of subtle bugs on real-world bibliography files.
"""


@dataclass(frozen=True)
class BibEntry:
    cite_key: str
    title: str
    authors: str
    year: int | None
    doi: str | None
    journal: str | None
    entry_type: str = ""
    booktitle: str | None = None
    note: str | None = None


@dataclass(frozen=True)
class ManuscriptSection:
    name: str
    text: str
    cite_keys: list[str]


_CITE_RE = re.compile(r"\\cite[a-zA-Z]*\{([^}]*)\}")
_SECTION_RE = re.compile(r"\\section\*?\{([^}]*)\}")
_COMMENT_RE = re.compile(r"(?<!\\)%.*")
_LATEX_COMMAND_RE = re.compile(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?(\{[^}]*\})?")


def parse_bibliography(bib_path: str | Path) -> dict[str, BibEntry]:
    with open(bib_path, encoding="utf-8") as handle:
        db = bibtexparser.load(handle)

    entries: dict[str, BibEntry] = {}
    for raw in db.entries:
        year_raw = raw.get("year")
        try:
            year = int(year_raw) if year_raw else None
        except ValueError:
            year = None
        entries[raw["ID"]] = BibEntry(
            cite_key=raw["ID"],
            title=(raw.get("title") or "").strip("{}"),
            authors=raw.get("author", ""),
            year=year,
            doi=raw.get("doi"),
            journal=raw.get("journal"),
            entry_type=raw.get("ENTRYTYPE", ""),
            booktitle=raw.get("booktitle"),
            note=raw.get("note"),
        )
    return entries


def _strip_latex(text: str) -> str:
    text = _COMMENT_RE.sub("", text)
    text = _LATEX_COMMAND_RE.sub(" ", text)
    text = text.replace("{", " ").replace("}", " ")
    return re.sub(r"\s+", " ", text).strip()


def parse_manuscript_sections(tex_path: str | Path) -> list[ManuscriptSection]:
    """Split on top-level \\section{}/\\section*{} boundaries and extract
    readable text plus citation keys used within each section."""
    raw = Path(tex_path).read_text(encoding="utf-8")
    doc_match = re.search(r"\\begin\{document\}(.*)\\end\{document\}", raw, re.DOTALL)
    body = doc_match.group(1) if doc_match else raw
    # Strip comments before locating sections, not just when stripping text
    # per-chunk - otherwise a commented-out `%\section{...}` (confirmed
    # present in the real wiley/revised_main.tex, line 746) still matches and
    # produces a spurious empty section.
    body = "\n".join(_COMMENT_RE.sub("", line) for line in body.splitlines())

    matches = list(_SECTION_RE.finditer(body))
    sections: list[ManuscriptSection] = []
    for i, match in enumerate(matches):
        name = match.group(1).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        chunk = body[start:end]
        cite_keys = sorted({
            key.strip()
            for cite_match in _CITE_RE.finditer(chunk)
            for key in cite_match.group(1).split(",")
            if key.strip()
        })
        sections.append(ManuscriptSection(name=name, text=_strip_latex(chunk), cite_keys=cite_keys))
    return sections


def find_manuscript_tex(directory: str | Path) -> Path | None:
    """Best-effort main-manuscript file selection.

    Prefers revised_main.tex over main.tex (a "revised" file is the newer,
    post-review version - confirmed by mtime on the real wiley/ folder this
    was built against). Falls back to the largest .tex file that isn't a
    latexdiff artifact (diff.tex).
    """
    directory = Path(directory)
    for candidate in ("revised_main.tex", "main.tex"):
        path = directory / candidate
        if path.exists():
            return path
    tex_files = [p for p in directory.glob("*.tex") if not p.name.startswith("diff")]
    if not tex_files:
        return None
    return max(tex_files, key=lambda p: p.stat().st_size)


def find_bibliography(directory: str | Path) -> Path | None:
    bib_files = list(Path(directory).glob("*.bib"))
    if not bib_files:
        return None
    return max(bib_files, key=lambda p: p.stat().st_size)
