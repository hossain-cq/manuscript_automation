from __future__ import annotations

"""LangGraph assessment + literature-grounding implementation.

The module is intentionally self-contained at the orchestration layer. Replace
ArtifactRepository, EvidenceRepository, and model credentials with production
implementations. The graph state contains IDs and compact records; large files,
PDFs, and full project artifacts must remain in an external artifact store.
"""

import hashlib
import json
import os
import re
import time
import uuid
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Annotated, Any, Literal, TypedDict

import requests
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from pathlib import Path

from ...settings import get_settings
from ...tools.model_gateway import (
    default_extra_body,
    default_max_completion_tokens,
    get_openai_client,
    strict_json_schema,
)


# ---------------------------------------------------------------------------
# Common utilities
# ---------------------------------------------------------------------------


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


_ARXIV_ATOM_NS = "{http://www.w3.org/2005/Atom}"
_ARXIV_EXT_NS = "{http://arxiv.org/schemas/atom}"


def parse_arxiv_feed(xml_text: str) -> list[dict[str, Any]]:
    """Parse an arXiv API Atom feed into plain dicts. arXiv's API is XML, not
    JSON like OpenAlex/Crossref - stdlib ElementTree is enough, no new
    dependency needed."""
    root = ET.fromstring(xml_text)
    entries: list[dict[str, Any]] = []
    for entry in root.findall(f"{_ARXIV_ATOM_NS}entry"):
        title_el = entry.find(f"{_ARXIV_ATOM_NS}title")
        id_el = entry.find(f"{_ARXIV_ATOM_NS}id")
        summary_el = entry.find(f"{_ARXIV_ATOM_NS}summary")
        published_el = entry.find(f"{_ARXIV_ATOM_NS}published")
        doi_el = entry.find(f"{_ARXIV_EXT_NS}doi")
        authors = [
            name_el.text.strip()
            for author_el in entry.findall(f"{_ARXIV_ATOM_NS}author")
            if (name_el := author_el.find(f"{_ARXIV_ATOM_NS}name")) is not None and name_el.text
        ]
        entries.append({
            "id": (id_el.text or "").strip() if id_el is not None else "",
            "title": re.sub(r"\s+", " ", (title_el.text or "").strip()) if title_el is not None else "",
            "summary": re.sub(r"\s+", " ", (summary_el.text or "").strip()) if summary_el is not None else None,
            "published": (published_el.text or "").strip() if published_el is not None else None,
            "authors": authors,
            "doi": (doi_el.text or "").strip() if doi_el is not None else None,
        })
    return entries


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def word_tokens(text: str) -> set[str]:
    stop = {
        "the", "and", "for", "with", "that", "this", "from", "using",
        "into", "based", "are", "was", "were", "has", "have", "our",
        "their", "than", "between", "over", "under", "show", "shows",
    }
    return {w for w in re.findall(r"[a-zA-Z0-9][a-zA-Z0-9+\-]{2,}", text.lower()) if w not in stop}


def lexical_similarity(a: str, b: str) -> float:
    left, right = word_tokens(a), word_tokens(b)
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


# ---------------------------------------------------------------------------
# Domain schemas
# ---------------------------------------------------------------------------

AssessmentLabel = Literal[
    "STRONGLY_SUPPORTED",
    "SUPPORTED_WITH_LIMITATIONS",
    "WEAKLY_SUPPORTED",
    "UNSUPPORTED",
    "NOT_ASSESSABLE",
]

ReadinessStatus = Literal[
    "READY_FOR_MANUSCRIPT",
    "DRAFTABLE_WITH_WARNINGS",
    "NEEDS_ADDITIONAL_ANALYSIS",
    "NEEDS_RESEARCH_COMPLETION",
    "INSUFFICIENT_EVIDENCE",
    "BLOCKED",
]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CandidateClaim(StrictModel):
    claim_id: str
    text: str
    claim_type: Literal[
        "RESULT", "METHOD", "INTERPRETATION", "HYPOTHESIS", "LITERATURE_CLAIM"
    ]
    importance: Literal["CENTRAL", "SUPPORTING", "CONTEXTUAL"]


class EvidenceItem(StrictModel):
    evidence_id: str
    source_artifact_id: str
    evidence_type: Literal[
        "USER_INPUT", "EXPERIMENTAL_RESULT", "COMPUTATIONAL_RESULT", "LITERATURE",
        "FIGURE", "TABLE", "CODE", "INFERENCE"
    ]
    location: str
    excerpt_or_value: str
    units: str | None = None
    extraction_confidence: float = Field(ge=0.0, le=1.0)


class EmpiricalFeature(StrictModel):
    feature_id: str
    value: float | None
    status: Literal["VALID", "MISSING", "NOT_APPLICABLE", "INVALID"]
    method: str
    input_artifact_ids: list[str] = Field(default_factory=list)
    finding_ids: list[str] = Field(default_factory=list)
    notes: str | None = None


class Finding(StrictModel):
    finding_id: str
    severity: Literal["INFO", "LOW", "MEDIUM", "HIGH", "CRITICAL"]
    rule_id: str
    message: str
    affected_claim_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    remediation: str
    blocking: bool = False


class ContributionCandidate(StrictModel):
    contribution_id: str
    statement: str
    contribution_type: Literal[
        "METHOD", "DATASET", "MATERIAL_OR_SYSTEM", "BENCHMARK",
        "MECHANISTIC_INSIGHT", "APPLICATION", "NEGATIVE_RESULT", "OTHER"
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


class ClaimAssessment(StrictModel):
    claim_id: str
    label: AssessmentLabel
    score: float = Field(ge=0.0, le=1.0)
    evidence_ids: list[str]
    reasoning: str
    limitations: list[str]
    missing_evidence: list[str]
    contradiction_finding_ids: list[str]
    confidence: float = Field(ge=0.0, le=1.0)


class LLMReadinessAssessment(StrictModel):
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


class VerifiedLiteratureRecord(StrictModel):
    literature_id: str
    provider: Literal["OPENALEX", "CROSSREF", "ARXIV"]
    provider_id: str
    title: str
    authors: list[str]
    year: int | None
    doi: str | None
    journal: str | None
    abstract: str | None
    source_url: str
    metadata_verified: bool
    verification_notes: list[str]
    query: str
    retrieved_at: str


class NoveltyComparison(StrictModel):
    claim_id: str
    literature_id: str
    similarity_type: Literal[
        "METHOD_SIMILARITY", "RESULT_SIMILARITY", "DATASET_SIMILARITY",
        "PROBLEM_SIMILARITY", "APPLICATION_SIMILARITY", "LOW_RELEVANCE"
    ]
    relationship: Literal[
        "EXTENDS", "REPRODUCES", "CONTRADICTS", "COMBINES", "DIFFERS",
        "UNCLEAR", "NOT_COMPARABLE"
    ]
    overlap_summary: str
    difference_summary: str
    evidence_ids: list[str]
    literature_ids: list[str]
    confidence: float = Field(ge=0.0, le=1.0)
    requires_human_review: bool


class NoveltyAssessment(StrictModel):
    claim_id: str
    novelty_status: Literal[
        "DISTINCT", "PARTIALLY_DISTINCT", "NOT_DISTINCT", "UNCLEAR", "NOT_ASSESSABLE"
    ]
    confidence: float = Field(ge=0.0, le=1.0)
    comparison_ids: list[str]
    reasoning: str
    missing_literature: list[str]
    requires_human_review: bool


class EvidencePacket(StrictModel):
    project_id: str
    domain_profile_id: str
    claims: list[CandidateClaim]
    evidence: list[EvidenceItem]
    empirical_features: list[EmpiricalFeature]
    findings: list[Finding]
    literature_records: list[VerifiedLiteratureRecord] = Field(default_factory=list)


class AssessmentReport(StrictModel):
    report_id: str
    project_id: str
    readiness_status: ReadinessStatus
    dimension_scores: dict[str, float]
    confidence_by_dimension: dict[str, float]
    central_claim_coverage: float
    llm_disagreement: float
    blocking_finding_ids: list[str]
    warning_finding_ids: list[str]
    contribution_candidates: list[ContributionCandidate]
    recommended_next_action: Literal[
        "PROCEED_TO_MANUSCRIPT_PLANNING",
        "DRAFT_WITH_WARNINGS",
        "CREATE_RESEARCH_COMPLETION_PLAN",
        "REQUEST_MORE_PROJECT_INFORMATION",
        "BLOCK_UNTIL_RESOLVED",
    ]
    explanation: str


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------

class AssessmentState(TypedDict, total=False):
    project_id: str
    packet_id: str
    packet: dict[str, Any]
    feature_ids: list[str]
    finding_ids: Annotated[list[str], list.__add__]
    blocking_finding_ids: list[str]
    assessor_ids: list[str]
    llm_assessment_ids: Annotated[list[str], list.__add__]
    llm_assessments: list[dict[str, Any]]
    dimension_scores: dict[str, float]
    confidence_by_dimension: dict[str, float]
    llm_disagreement: float
    report_id: str
    report: dict[str, Any]
    human_decision: dict[str, Any]
    error: str | None


class NoveltyState(TypedDict, total=False):
    project_id: str
    packet_id: str
    packet: dict[str, Any]
    query_by_claim: dict[str, list[str]]
    literature_records: Annotated[list[dict[str, Any]], list.__add__]
    comparison_ids: Annotated[list[str], list.__add__]
    comparisons: Annotated[list[dict[str, Any]], list.__add__]
    novelty_assessments: Annotated[list[dict[str, Any]], list.__add__]
    finding_ids: Annotated[list[str], list.__add__]
    report_id: str
    human_decision: dict[str, Any]
    error: str | None


# ---------------------------------------------------------------------------
# Persistence interfaces
# ---------------------------------------------------------------------------

class InMemoryRepository:
    """Replace with a database/artifact repository in production."""

    def __init__(self) -> None:
        self.features: dict[str, EmpiricalFeature] = {}
        self.findings: dict[str, Finding] = {}
        self.assessments: dict[str, LLMReadinessAssessment] = {}
        self.literature: dict[str, VerifiedLiteratureRecord] = {}
        self.comparisons: dict[str, NoveltyComparison] = {}
        self.novelty: dict[str, NoveltyAssessment] = {}
        self.reports: dict[str, AssessmentReport] = {}

    def add_feature(self, item: EmpiricalFeature) -> str:
        self.features[item.feature_id] = item
        return item.feature_id

    def add_finding(self, item: Finding) -> str:
        self.findings[item.finding_id] = item
        return item.finding_id

    def add_assessment(self, item: LLMReadinessAssessment) -> str:
        key = f"{item.assessor_id}-{new_id('A')}"
        self.assessments[key] = item
        return key

    def add_literature(self, item: VerifiedLiteratureRecord) -> str:
        self.literature[item.literature_id] = item
        return item.literature_id

    def add_comparison(self, item: NoveltyComparison) -> str:
        self.comparisons[item.claim_id + ':' + item.literature_id] = item
        return item.claim_id + ':' + item.literature_id

    def add_novelty(self, item: NoveltyAssessment) -> str:
        key = item.claim_id + ':' + new_id('N')
        self.novelty[key] = item
        return key

    def add_report(self, item: AssessmentReport) -> str:
        self.reports[item.report_id] = item
        return item.report_id


# ---------------------------------------------------------------------------
# Structured LLM gateway
# ---------------------------------------------------------------------------

class StructuredLLMGateway:
    """OpenAI-compatible structured-output gateway.

    The built-in proxy is configured through OPENAI_API_KEY and OPENAI_API_BASE.
    The response schema is strict and rejects extra properties.
    """

    def __init__(self, model: str = "gpt-5-mini") -> None:
        self.client = get_openai_client()
        self.model = model

    def call(self, *, system: str, user: str, schema: type[BaseModel]) -> BaseModel:
        schema_json = strict_json_schema(schema)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": schema.__name__.lower(),
                    "strict": True,
                    "schema": schema_json,
                },
            },
            max_completion_tokens=default_max_completion_tokens(8000),
            extra_body=default_extra_body(),
        )
        content = response.choices[0].message.content
        if not content:
            raise ValueError("LLM returned empty structured content")
        return schema.model_validate_json(content)


# ---------------------------------------------------------------------------
# Deterministic feature and blocker logic
# ---------------------------------------------------------------------------


def feature_id(name: str, project_id: str) -> str:
    return f"FEATURE-{project_id}-{name}"


def compute_empirical_features(packet: EvidencePacket, repo: InMemoryRepository) -> list[str]:
    claims = packet.claims
    evidence_ids = {item.evidence_id for item in packet.evidence}
    supported_claims = 0
    central_claims = [c for c in claims if c.importance == "CENTRAL"]

    # Conservative proxy: a claim counts as covered only when at least one
    # evidence item explicitly uses its claim ID in the location/metadata.
    for claim in central_claims:
        if any(claim.claim_id in item.location for item in packet.evidence):
            supported_claims += 1

    coverage_value = (
        supported_claims / len(central_claims) if central_claims else 0.0
    )
    coverage = EmpiricalFeature(
        feature_id=feature_id("central_claim_coverage", packet.project_id),
        value=coverage_value,
        status="VALID" if central_claims else "MISSING",
        method="explicit_claim_id_evidence_link_count_v1",
        input_artifact_ids=[item.source_artifact_id for item in packet.evidence],
        notes=f"{supported_claims}/{len(central_claims)} central claims covered",
    )
    repo.add_feature(coverage)

    provenance_value = (
        len(evidence_ids) / max(1, len(packet.evidence))
        if packet.evidence else 0.0
    )
    provenance = EmpiricalFeature(
        feature_id=feature_id("evidence_presence", packet.project_id),
        value=provenance_value,
        status="VALID" if packet.evidence else "MISSING",
        method="evidence_registry_presence_v1",
        input_artifact_ids=[item.source_artifact_id for item in packet.evidence],
    )
    repo.add_feature(provenance)

    feature_ids = [coverage.feature_id, provenance.feature_id]
    for supplied in packet.empirical_features:
        repo.add_feature(supplied)
        feature_ids.append(supplied.feature_id)
    return feature_ids


def evaluate_blockers(packet: EvidencePacket, repo: InMemoryRepository) -> list[str]:
    finding_ids: list[str] = []
    central = [c for c in packet.claims if c.importance == "CENTRAL"]
    evidence_text = " ".join(item.location for item in packet.evidence)

    for claim in central:
        if claim.claim_id not in evidence_text:
            finding = Finding(
                finding_id=new_id("FINDING"),
                severity="HIGH",
                rule_id="central_claim_without_explicit_evidence_link",
                message=f"Central claim {claim.claim_id} has no explicit evidence link.",
                affected_claim_ids=[claim.claim_id],
                remediation="Link the claim to direct evidence or downgrade/remove the claim.",
                blocking=True,
            )
            repo.add_finding(finding)
            finding_ids.append(finding.finding_id)

    for supplied in packet.findings:
        repo.add_finding(supplied)
        finding_ids.append(supplied.finding_id)
    return finding_ids


def packet_text(packet: EvidencePacket) -> str:
    return json.dumps(packet.model_dump(mode="json"), indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Literature providers
# ---------------------------------------------------------------------------

@dataclass
class LiteratureQuery:
    claim_id: str
    query: str


class LiteratureClient:
    """Metadata retrieval using public scholarly metadata providers.

    Search results are candidates only. Records become verified only after
    normalizing provider IDs and validating the available metadata.
    """

    def __init__(self, timeout: int = 20, cache_dir: str | Path | None = None) -> None:
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "scientific-manuscript-system/0.1"})
        # Un-expiring, on-disk cache keyed by (provider, query, params). No
        # TTL: an already-published work's OpenAlex/Crossref record doesn't
        # meaningfully change. Added after a real run burned ~2 minutes
        # re-querying all 70 citations in a bibliography that hadn't
        # changed - every re-run paid the same rate-limit risk for nothing.
        # Delete the cache directory by hand to force fresh lookups.
        self.cache_dir = Path(cache_dir) if cache_dir else Path(get_settings().artifact_store_path).parent / "literature_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    _NEGATIVE_CACHE_SECONDS = 300  # see _cached_get docstring

    def _cache_path(self, provider: str, query: str, **params: Any) -> Path:
        key_material = json.dumps({"provider": provider, "query": query, **params}, sort_keys=True)
        digest = hashlib.sha256(key_material.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.json"

    def _get_with_retry(self, url: str, *, params: dict[str, Any], max_retries: int = 2) -> requests.Response:
        """Retry on 429 (rate limit) with backoff, honoring Retry-After when
        present. Anonymous OpenAlex/Crossref requests share a low rate limit;
        confirmed by hand that a burst of ~70 sequential citation-verification
        searches (manuscript_evaluation.py) triggers 429s well within that
        burst - without this, search_verified()'s bare `except
        requests.RequestException: pass` turns a rate-limited request into an
        indistinguishable false "not found"."""
        delay = 1.0
        for attempt in range(max_retries + 1):
            response = self.session.get(url, params=params, timeout=self.timeout)
            if response.status_code != 429 or attempt == max_retries:
                response.raise_for_status()
                return response
            wait = float(response.headers.get("Retry-After", delay))
            time.sleep(min(wait, 5.0))
            delay *= 2
        raise AssertionError("unreachable")

    def _cached_get(self, provider: str, url: str, *, query: str, **params: Any) -> list[dict[str, Any]]:
        """Cache both successes (no TTL - a published record doesn't change)
        and failures (short TTL). Without negative caching, a provider that's
        persistently rate-limited - confirmed against OpenAlex after enough
        testing today to exhaust its anonymous quota - has nothing to write
        to the success cache, so _get_with_retry's full ~10s backoff sequence
        replays on every single run, for every affected query, forever. 5
        minutes is short enough to recover once the limit resets, long
        enough that one evaluation run (a minute or two) doesn't re-pay it
        per citation."""
        cache_path = self._cache_path(provider, query, **params)
        if cache_path.exists():
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            if payload.get("ok"):
                return payload["results"]
            if time.time() - payload.get("failed_at", 0) < self._NEGATIVE_CACHE_SECONDS:
                return []
        try:
            response = self._get_with_retry(url, params=params)
        except requests.RequestException:
            cache_path.write_text(json.dumps({"ok": False, "failed_at": time.time()}), encoding="utf-8")
            return []
        results = self._extract_results(provider, response)
        cache_path.write_text(json.dumps({"ok": True, "results": results}), encoding="utf-8")
        return results

    @staticmethod
    def _extract_results(provider: str, response: requests.Response) -> list[dict[str, Any]]:
        if provider == "openalex":
            return response.json().get("results", [])
        if provider == "arxiv":
            return parse_arxiv_feed(response.text)
        return response.json().get("message", {}).get("items", [])

    def openalex_search(self, query: str, per_page: int = 10) -> list[dict[str, Any]]:
        return self._cached_get(
            "openalex", "https://api.openalex.org/works", query=query, search=query, **{"per-page": per_page}
        )

    def crossref_search(self, query: str, rows: int = 10) -> list[dict[str, Any]]:
        return self._cached_get(
            "crossref", "https://api.crossref.org/works", query=query, **{"query.bibliographic": query, "rows": rows}
        )

    def arxiv_search(self, query: str, max_results: int = 5) -> list[dict[str, Any]]:
        # ti:"..." searches the title field specifically - our own
        # lexical_similarity() scoring on the returned candidates does the
        # real matching, this just narrows the candidate set.
        return self._cached_get(
            "arxiv", "https://export.arxiv.org/api/query", query=query,
            **{"search_query": f'ti:"{query}"', "max_results": max_results},
        )

    @staticmethod
    def reconstruct_openalex_abstract(record: dict[str, Any]) -> str | None:
        inverted = record.get("abstract_inverted_index")
        if not inverted:
            return None
        tokens: list[tuple[int, str]] = []
        for word, positions in inverted.items():
            tokens.extend((position, word) for position in positions)
        return " ".join(word for _, word in sorted(tokens))

    def normalize_openalex(
        self, item: dict[str, Any], query: str
    ) -> VerifiedLiteratureRecord | None:
        title = (item.get("title") or "").strip()
        if not title:
            return None
        doi = item.get("doi")
        doi = doi.replace("https://doi.org/", "") if doi else None
        authors = [
            (a.get("author") or {}).get("display_name", "")
            for a in item.get("authorships", [])
        ]
        authors = [name for name in authors if name]
        year = item.get("publication_year")
        source = item.get("primary_location", {}).get("source") or {}
        journal = source.get("display_name")
        return VerifiedLiteratureRecord(
            literature_id=f"OPENALEX:{item.get('id', new_id('OA'))}",
            provider="OPENALEX",
            provider_id=item.get("id", ""),
            title=title,
            authors=authors,
            year=year,
            doi=doi,
            journal=journal,
            abstract=self.reconstruct_openalex_abstract(item),
            source_url=item.get("doi") or item.get("id") or "",
            metadata_verified=bool(item.get("id")),
            verification_notes=["Metadata retrieved from OpenAlex"],
            query=query,
            retrieved_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )

    def normalize_crossref(
        self, item: dict[str, Any], query: str
    ) -> VerifiedLiteratureRecord | None:
        title_values = item.get("title") or []
        title = title_values[0].strip() if title_values else ""
        if not title:
            return None
        authors = []
        for author in item.get("author", []):
            name = " ".join(
                part for part in [author.get("given"), author.get("family")] if part
            )
            if name:
                authors.append(name)
        doi = item.get("DOI")
        return VerifiedLiteratureRecord(
            literature_id=f"CROSSREF:{doi or item.get('URL', new_id('CR'))}",
            provider="CROSSREF",
            provider_id=doi or item.get("URL", ""),
            title=title,
            authors=authors,
            year=(item.get("published-print") or item.get("published-online") or {})
            .get("date-parts", [[None]])[0][0],
            doi=doi,
            journal=(item.get("container-title") or [None])[0],
            abstract=None,
            source_url=item.get("URL") or (f"https://doi.org/{doi}" if doi else ""),
            metadata_verified=bool(doi or item.get("URL")),
            verification_notes=["Metadata retrieved from Crossref"],
            query=query,
            retrieved_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )

    def normalize_arxiv(self, item: dict[str, Any], query: str) -> VerifiedLiteratureRecord | None:
        title = item.get("title") or ""
        if not title:
            return None
        arxiv_id = item.get("id", "")
        year = None
        published = item.get("published")
        if published:
            try:
                year = int(published[:4])
            except ValueError:
                year = None
        return VerifiedLiteratureRecord(
            literature_id=f"ARXIV:{arxiv_id or new_id('AX')}",
            provider="ARXIV",
            provider_id=arxiv_id,
            title=title,
            authors=item.get("authors", []),
            year=year,
            doi=item.get("doi"),
            journal=None,
            abstract=item.get("summary"),
            source_url=arxiv_id,
            metadata_verified=bool(arxiv_id),
            verification_notes=["Metadata retrieved from arXiv"],
            query=query,
            retrieved_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )

    def search_verified(self, query: LiteratureQuery, limit: int = 8) -> list[VerifiedLiteratureRecord]:
        records: list[VerifiedLiteratureRecord] = []
        try:
            for item in self.openalex_search(query.query, per_page=limit):
                normalized = self.normalize_openalex(item, query.query)
                if normalized:
                    records.append(normalized)
        except requests.RequestException:
            pass
        try:
            for item in self.crossref_search(query.query, rows=limit):
                normalized = self.normalize_crossref(item, query.query)
                if normalized:
                    records.append(normalized)
        except requests.RequestException:
            pass

        # Deduplicate by DOI first, then normalized title. Never merge two
        # records merely because an LLM believes they are the same paper.
        deduped: dict[str, VerifiedLiteratureRecord] = {}
        for record in records:
            key = (record.doi or re.sub(r"\W+", "", record.title.lower()))
            if key not in deduped:
                deduped[key] = record
        return list(deduped.values())[:limit]


# ---------------------------------------------------------------------------
# Prompting and literature novelty logic
# ---------------------------------------------------------------------------

READINESS_SYSTEM = """You are a conservative scientific assessment agent.
Use only the evidence IDs and finding IDs in the supplied packet. Never invent
references, numbers, experiments, or metadata. Evaluate evidence readiness and
potential significance separately. If the packet is incomplete, set abstain=true.
Every claim assessment must cite evidence IDs. Distinguish result from
interpretation and candidate novelty from established novelty."""

NOVELTY_SYSTEM = """You are a conservative literature-comparison agent.
You may compare the candidate claim only with the supplied verified literature
records. Do not claim novelty or priority. Report similarities, differences,
limitations, and uncertainty. If abstracts or metadata are insufficient, use
UNCLEAR or NOT_ASSESSABLE and require human review. Use only supplied IDs."""


def build_assessor_prompt(packet: EvidencePacket, assessor_id: str) -> str:
    return json.dumps({
        "assessor_id": assessor_id,
        "instructions": [
            "Score each dimension from 0 to 1.",
            "Return only the requested structured schema.",
            "Cite supplied evidence and finding IDs.",
            "Do not infer missing experimental or computational results.",
        ],
        "packet": packet.model_dump(mode="json"),
    }, ensure_ascii=False)


def build_novelty_prompt(
    claim: CandidateClaim,
    records: list[VerifiedLiteratureRecord],
    evidence: list[EvidenceItem],
) -> str:
    return json.dumps({
        "claim": claim.model_dump(mode="json"),
        "literature_records": [r.model_dump(mode="json") for r in records],
        "supporting_evidence": [e.model_dump(mode="json") for e in evidence],
        "instructions": [
            "Compare the candidate claim with each record where relevant.",
            "Do not call a claim novel merely because no similar record was retrieved.",
            "Use UNCLEAR when search coverage or metadata is insufficient.",
        ],
    }, ensure_ascii=False)


def compare_assessments(assessments: list[LLMReadinessAssessment]) -> tuple[dict[str, float], dict[str, float], float]:
    if not assessments:
        raise ValueError("No LLM assessments available")
    dimensions = [
        "scientific_contribution", "evidence_sufficiency", "methodological_rigor",
        "validation_strength", "reproducibility", "literature_positioning",
        "potential_significance",
    ]
    scores: dict[str, float] = {}
    confidences: dict[str, float] = {}
    for dimension in dimensions:
        values = [getattr(a, dimension) for a in assessments if not a.abstain]
        if not values:
            scores[dimension] = 0.0
            confidences[dimension] = 0.0
            continue
        scores[dimension] = sum(values) / len(values)
        mean = scores[dimension]
        disagreement = sum(abs(value - mean) for value in values) / len(values)
        confidences[dimension] = clamp01(1.0 - disagreement)
    all_scores = [
        getattr(a, dimension)
        for a in assessments
        if not a.abstain
        for dimension in dimensions
    ]
    global_disagreement = 0.0
    if all_scores:
        global_mean = sum(all_scores) / len(all_scores)
        global_disagreement = sum(abs(x - global_mean) for x in all_scores) / len(all_scores)
    return scores, confidences, clamp01(global_disagreement)


def determine_readiness(
    packet: EvidencePacket,
    findings: list[Finding],
    scores: dict[str, float],
    disagreement: float,
) -> tuple[ReadinessStatus, str]:
    blocking = [f for f in findings if f.blocking or f.severity == "CRITICAL"]
    coverage = next(
        (f.value for f in packet.empirical_features if f.feature_id.endswith("central_claim_coverage")),
        0.0,
    )
    if blocking:
        return "BLOCKED", "Blocking scientific-integrity or evidence findings remain."
    if coverage < 0.70:
        return "INSUFFICIENT_EVIDENCE", "Central-claim evidence coverage is below policy threshold."
    if scores.get("evidence_sufficiency", 0.0) < 0.60 or scores.get("validation_strength", 0.0) < 0.55:
        return "NEEDS_ADDITIONAL_ANALYSIS", "Evidence or validation strength is insufficient for the current claims."
    if disagreement > 0.25:
        return "DRAFTABLE_WITH_WARNINGS", "Independent assessors disagree materially; human review is required."
    if scores.get("reproducibility", 0.0) < 0.50:
        return "DRAFTABLE_WITH_WARNINGS", "A manuscript may be drafted, but reproducibility documentation is incomplete."
    return "READY_FOR_MANUSCRIPT", "No critical blockers were detected under the active policy profile."


# ---------------------------------------------------------------------------
# Assessment graph nodes
# ---------------------------------------------------------------------------


def load_packet_node(state: AssessmentState) -> dict[str, Any]:
    packet = EvidencePacket.model_validate(state["packet"])
    return {"packet": packet.model_dump(mode="json"), "error": None}


def compute_features_node(state: AssessmentState, repo: InMemoryRepository) -> dict[str, Any]:
    packet = EvidencePacket.model_validate(state["packet"])
    ids = compute_empirical_features(packet, repo)
    return {"feature_ids": ids}


def blockers_node(state: AssessmentState, repo: InMemoryRepository) -> dict[str, Any]:
    packet = EvidencePacket.model_validate(state["packet"])
    ids = evaluate_blockers(packet, repo)
    blocking = [repo.findings[i].finding_id for i in ids if repo.findings[i].blocking]
    return {"finding_ids": ids, "blocking_finding_ids": blocking}


def assessor_node(
    state: AssessmentState,
    repo: InMemoryRepository,
    gateway: StructuredLLMGateway,
    assessor_id: str,
) -> dict[str, Any]:
    packet = EvidencePacket.model_validate(state["packet"])
    result = gateway.call(
        system=READINESS_SYSTEM,
        user=build_assessor_prompt(packet, assessor_id),
        schema=LLMReadinessAssessment,
    )
    if not result.evidence_ids and packet.claims:
        result = result.model_copy(update={"abstain": True})
    result = result.model_copy(update={"assessor_id": assessor_id})
    assessment_id = repo.add_assessment(result)
    return {
        "llm_assessment_ids": [assessment_id],
        "llm_assessments": [result.model_dump(mode="json")],
    }


def aggregate_node(state: AssessmentState, repo: InMemoryRepository) -> dict[str, Any]:
    packet = EvidencePacket.model_validate(state["packet"])
    assessments = [LLMReadinessAssessment.model_validate(x) for x in state.get("llm_assessments", [])]
    if not assessments:
        raise ValueError("No valid LLM assessments")
    scores, confidence, disagreement = compare_assessments(assessments)
    findings = [repo.findings[x] for x in state.get("finding_ids", []) if x in repo.findings]
    status, explanation = determine_readiness(packet, findings, scores, disagreement)
    report = AssessmentReport(
        report_id=new_id("ASSESSMENT"),
        project_id=state["project_id"],
        readiness_status=status,
        dimension_scores=scores,
        confidence_by_dimension=confidence,
        central_claim_coverage=next(
            (f.value or 0.0 for f in packet.empirical_features if f.feature_id.endswith("central_claim_coverage")),
            0.0,
        ),
        llm_disagreement=disagreement,
        blocking_finding_ids=state.get("blocking_finding_ids", []),
        warning_finding_ids=[f.finding_id for f in findings if f.severity in {"LOW", "MEDIUM", "HIGH"} and not f.blocking],
        contribution_candidates=[c for a in assessments for c in a.contribution_candidates],
        recommended_next_action=(
            "BLOCK_UNTIL_RESOLVED" if status == "BLOCKED" else
            "CREATE_RESEARCH_COMPLETION_PLAN" if status in {"NEEDS_ADDITIONAL_ANALYSIS", "INSUFFICIENT_EVIDENCE"} else
            "DRAFT_WITH_WARNINGS" if status == "DRAFTABLE_WITH_WARNINGS" else
            "PROCEED_TO_MANUSCRIPT_PLANNING"
        ),
        explanation=explanation,
    )
    repo.add_report(report)
    return {
        "dimension_scores": scores,
        "confidence_by_dimension": confidence,
        "llm_disagreement": disagreement,
        "report_id": report.report_id,
        "report": report.model_dump(mode="json"),
    }


def human_review_node(state: AssessmentState) -> dict[str, Any]:
    response = interrupt({
        "kind": "PUBLISHABILITY_ASSESSMENT_REVIEW",
        "report_id": state["report_id"],
        "readiness_status": state["report"]["readiness_status"],
        "dimension_scores": state["dimension_scores"],
        "blocking_finding_ids": state.get("blocking_finding_ids", []),
        "choices": [
            "APPROVE_MANUSCRIPT_PLANNING",
            "APPROVE_COMPLETION_PLAN",
            "REQUEST_REASSESSMENT",
            "BLOCK_RUN",
        ],
    })
    decision = response.get("decision") if isinstance(response, dict) else response
    allowed = {
        "APPROVE_MANUSCRIPT_PLANNING",
        "APPROVE_COMPLETION_PLAN",
        "REQUEST_REASSESSMENT",
        "BLOCK_RUN",
    }
    if decision not in allowed:
        raise ValueError(f"Invalid assessment decision: {decision!r}")
    return {"human_decision": {"decision": decision}}


def build_assessment_graph(
    repo: InMemoryRepository,
    gateway: StructuredLLMGateway,
):
    builder = StateGraph(AssessmentState)
    builder.add_node("load_packet", load_packet_node)
    builder.add_node("compute_features", lambda s: compute_features_node(s, repo))
    builder.add_node("evaluate_blockers", lambda s: blockers_node(s, repo))
    builder.add_node("assess_contribution", lambda s: assessor_node(s, repo, gateway, "contribution_assessor"))
    builder.add_node("assess_evidence", lambda s: assessor_node(s, repo, gateway, "evidence_assessor"))
    builder.add_node("assess_critical_review", lambda s: assessor_node(s, repo, gateway, "critical_reviewer"))
    builder.add_node("aggregate", lambda s: aggregate_node(s, repo))
    builder.add_node("human_review", human_review_node)

    builder.add_edge(START, "load_packet")
    builder.add_edge("load_packet", "compute_features")
    builder.add_edge("compute_features", "evaluate_blockers")
    # These edges create a fan-out. State reducers accumulate assessment IDs.
    builder.add_edge("evaluate_blockers", "assess_contribution")
    builder.add_edge("evaluate_blockers", "assess_evidence")
    builder.add_edge("evaluate_blockers", "assess_critical_review")
    builder.add_edge("assess_contribution", "aggregate")
    builder.add_edge("assess_evidence", "aggregate")
    builder.add_edge("assess_critical_review", "aggregate")
    builder.add_edge("aggregate", "human_review")
    builder.add_edge("human_review", END)
    return builder.compile()


# ---------------------------------------------------------------------------
# Novelty and literature graph nodes
# ---------------------------------------------------------------------------


def make_queries_node(state: NoveltyState) -> dict[str, Any]:
    packet = EvidencePacket.model_validate(state["packet"])
    query_by_claim: dict[str, list[str]] = {}
    for claim in packet.claims:
        # Use compact query variants. The title/abstract retriever should not
        # use unbounded prose from the whole project.
        terms = sorted(word_tokens(claim.text), key=len, reverse=True)[:10]
        base = " ".join(terms)
        query_by_claim[claim.claim_id] = [
            base,
            f"{base} method",
            f"{base} benchmark mechanism",
        ]
    return {"query_by_claim": query_by_claim}


def retrieve_literature_node(
    state: NoveltyState,
    repo: InMemoryRepository,
    literature_client: LiteratureClient,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    packet = EvidencePacket.model_validate(state["packet"])
    for claim in packet.claims:
        for query in state["query_by_claim"].get(claim.claim_id, []):
            for record in literature_client.search_verified(LiteratureQuery(claim.claim_id, query)):
                repo.add_literature(record)
                records.append(record.model_dump(mode="json"))
    # Deduplicate in graph update too, because multiple query variants overlap.
    unique: dict[str, dict[str, Any]] = {}
    for record in records:
        unique[record["literature_id"]] = record
    return {"literature_records": list(unique.values())}


def lexical_filter_node(state: NoveltyState) -> dict[str, Any]:
    packet = EvidencePacket.model_validate(state["packet"])
    records = [VerifiedLiteratureRecord.model_validate(x) for x in state.get("literature_records", [])]
    filtered: list[dict[str, Any]] = []
    for claim in packet.claims:
        scored = []
        for record in records:
            text = record.title + " " + (record.abstract or "")
            scored.append((lexical_similarity(claim.text, text), record))
        scored.sort(key=lambda x: x[0], reverse=True)
        filtered.extend(r.model_dump(mode="json") for score, r in scored[:8] if score >= 0.02)
    unique = {r["literature_id"]: r for r in filtered}
    return {"literature_records": list(unique.values())}


def novelty_comparison_node(
    state: NoveltyState,
    repo: InMemoryRepository,
    gateway: StructuredLLMGateway,
) -> dict[str, Any]:
    packet = EvidencePacket.model_validate(state["packet"])
    records = [VerifiedLiteratureRecord.model_validate(x) for x in state.get("literature_records", [])]
    comparisons: list[dict[str, Any]] = []
    novelty_assessments: list[dict[str, Any]] = []
    for claim in packet.claims:
        relevant = [
            r for r in records
            if claim.claim_id in r.query or lexical_similarity(claim.text, r.title + " " + (r.abstract or "")) >= 0.02
        ][:8]
        if not relevant:
            novelty_assessments.append(NoveltyAssessment(
                claim_id=claim.claim_id,
                novelty_status="NOT_ASSESSABLE",
                confidence=0.0,
                comparison_ids=[],
                reasoning="No verified literature records with sufficient metadata were retrieved.",
                missing_literature=["Broader search, domain terms, or expert-provided references are required."],
                requires_human_review=True,
            ).model_dump(mode="json"))
            continue
        response = gateway.call(
            system=NOVELTY_SYSTEM,
            user=build_novelty_prompt(claim, relevant, packet.evidence),
            schema=NoveltyComparison,
        )
        # The strict response represents one comparison. For several records,
        # call the model separately in production or use a list schema. Here we
        # verify that the returned literature ID belongs to the retrieved set.
        valid_ids = {r.literature_id for r in relevant}
        if response.literature_id not in valid_ids:
            raise ValueError("Novelty model returned literature ID outside packet")
        comparison_id = repo.add_comparison(response)
        comparisons.append(response.model_dump(mode="json"))
        novelty_assessments.append(NoveltyAssessment(
            claim_id=claim.claim_id,
            novelty_status=(
                "DISTINCT" if response.relationship in {"DIFFERS", "EXTENDS"} and response.confidence >= 0.75
                else "PARTIALLY_DISTINCT" if response.relationship in {"COMBINES", "EXTENDS"}
                else "NOT_DISTINCT" if response.relationship in {"REPRODUCES"}
                else "UNCLEAR"
            ),
            confidence=response.confidence,
            comparison_ids=[comparison_id],
            reasoning=response.difference_summary,
            missing_literature=[],
            requires_human_review=response.requires_human_review or response.confidence < 0.75,
        ).model_dump(mode="json"))
    return {
        "comparison_ids": [c["claim_id"] + ":" + c["literature_id"] for c in comparisons],
        "comparisons": comparisons,
        "novelty_assessments": novelty_assessments,
    }


def novelty_human_review_node(state: NoveltyState) -> dict[str, Any]:
    uncertain = [
        a for a in state.get("novelty_assessments", [])
        if a["requires_human_review"] or a["novelty_status"] in {"DISTINCT", "NOT_ASSESSABLE"}
    ]
    if not uncertain:
        return {"human_decision": {"decision": "AUTO_ACCEPT_LOW_RISK"}}
    decision = interrupt({
        "kind": "NOVELTY_REVIEW",
        "project_id": state["project_id"],
        "novelty_assessments": uncertain,
        "literature_records": state.get("literature_records", []),
        "choices": ["ACCEPT_AS_CANDIDATE", "REQUEST_MORE_SEARCH", "REJECT_COMPARISON"],
    })
    return {"human_decision": decision if isinstance(decision, dict) else {"decision": decision}}


def build_novelty_graph(
    repo: InMemoryRepository,
    gateway: StructuredLLMGateway,
    literature_client: LiteratureClient,
):
    builder = StateGraph(NoveltyState)
    builder.add_node("make_queries", make_queries_node)
    builder.add_node("retrieve_literature", lambda s: retrieve_literature_node(s, repo, literature_client))
    builder.add_node("lexical_filter", lexical_filter_node)
    builder.add_node("compare_novelty", lambda s: novelty_comparison_node(s, repo, gateway))
    builder.add_node("human_review", novelty_human_review_node)
    builder.add_edge(START, "make_queries")
    builder.add_edge("make_queries", "retrieve_literature")
    builder.add_edge("retrieve_literature", "lexical_filter")
    builder.add_edge("lexical_filter", "compare_novelty")
    builder.add_edge("compare_novelty", "human_review")
    builder.add_edge("human_review", END)
    return builder.compile()


# ---------------------------------------------------------------------------
# Example construction
# ---------------------------------------------------------------------------


def build_system():
    repo = InMemoryRepository()
    assessment_gateway = StructuredLLMGateway(model=os.getenv("ASSESSMENT_MODEL", "gpt-5-mini"))
    novelty_gateway = StructuredLLMGateway(model=os.getenv("NOVELTY_MODEL", "gpt-5"))
    literature_client = LiteratureClient()
    return (
        build_assessment_graph(repo, assessment_gateway),
        build_novelty_graph(repo, novelty_gateway, literature_client),
        repo,
    )


if __name__ == "__main__":
    print("Module loaded. Construct graphs with build_system().")
