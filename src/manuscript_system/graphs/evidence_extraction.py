from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from ..domain.enums import EvidenceType
from ..domain.models import Claim, Evidence, Finding, SourceAsset
from ..persistence.repositories import new_id
from ..settings import get_settings
from .subgraphs.novelty_and_publishability import CandidateClaim, EvidenceItem, EvidencePacket

"""Deterministic (no-LLM) domain classification and evidence extraction.

Turns the raw scanned files from create_manifest into structured Claim/
Evidence records, using each configs/domains/*.yaml profile's declarative
method_vocabulary and completeness_checklist rather than any hardcoded
Python heuristics. Never invents a claim with no matching file - a checklist
item with no match produces a non-blocking Finding instead.
"""

EXTENSION_EVIDENCE_TYPE: dict[str, EvidenceType] = {
    ".jpg": "FIGURE", ".jpeg": "FIGURE", ".png": "FIGURE",
    ".csv": "TABLE", ".dat": "TABLE",
    ".ipynb": "CODE", ".py": "CODE",
}

# Cap evidence items per claim rather than dumping every matching file. Real
# projects can match many files per checklist item (AQT_electrolyte matched
# up to 9 for one item); the packet gets sent whole to every LLM assessor
# call, and Groq's free tier enforces an 8000 tokens-per-minute request cap
# that a full evidence dump for a multi-claim project exceeds regardless of
# max_completion_tokens (confirmed: request rejected at 9081 tokens with
# max_completion_tokens=4000). A handful of representative files is enough
# grounding for a keyword-matched claim; this is a size cap, not a quality one.
MAX_EVIDENCE_PER_CLAIM = 3

# novelty_and_publishability.CandidateClaim.claim_type doesn't include FACT or
# SUGGESTION (domain.models.ClaimType does). Extraction only ever emits
# RESULT today, but map defensively so a future claim_type still translates.
_CLAIM_TYPE_MAP: dict[str, str] = {
    "FACT": "RESULT", "RESULT": "RESULT", "INTERPRETATION": "INTERPRETATION",
    "HYPOTHESIS": "HYPOTHESIS", "LITERATURE_CLAIM": "LITERATURE_CLAIM", "SUGGESTION": "INTERPRETATION",
}


def load_domain_profile(domain_id: str) -> dict[str, Any] | None:
    path = Path(get_settings().configs_path) / "domains" / f"{domain_id}.yaml"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return (data or {}).get("domain_profile")


def _list_domain_profiles() -> list[dict[str, Any]]:
    domains_dir = Path(get_settings().configs_path) / "domains"
    if not domains_dir.exists():
        return []
    profiles = []
    for path in sorted(domains_dir.glob("*.yaml")):
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        if data and "domain_profile" in data:
            profiles.append(data["domain_profile"])
    return profiles


def _fingerprint_words(vocabulary: list[str]) -> set[str]:
    """Split multi-word vocabulary phrases into matchable tokens.

    Keeps short uppercase acronyms (VQE, SQD) that a length filter alone
    would drop, while filtering generic short lowercase words (e.g. "set").
    """
    words: set[str] = set()
    for phrase in vocabulary:
        for token in re.split(r"[\s\-]+", phrase):
            if token and (len(token) >= 4 or token.isupper()):
                words.add(token.lower())
    return words


def classify_domain_from_assets(assets: list[SourceAsset]) -> tuple[str, float]:
    """Score every configs/domains/*.yaml profile against the scanned file
    paths and return the best match. Falls back to ("generic", 0.0) - no
    false positives on synthetic or empty projects."""
    profiles = _list_domain_profiles()
    if not profiles or not assets:
        return "generic", 0.0

    haystack = " ".join(asset.relative_path.lower() for asset in assets)
    best_id, best_confidence = "generic", 0.0
    for profile in profiles:
        words = _fingerprint_words(profile.get("method_vocabulary", []))
        if not words:
            continue
        matched = {w for w in words if w in haystack}
        confidence = len(matched) / len(words)
        if confidence > best_confidence:
            best_id, best_confidence = profile["domain_id"], confidence
    return best_id, best_confidence


def extract_evidence_and_claims(
    assets: list[SourceAsset],
    domain_profile: dict[str, Any] | None,
    project_id: str,
) -> tuple[list[Claim], list[Evidence], list[Finding]]:
    """For each completeness_checklist item, keyword-match scanned file
    paths. A match produces one Claim plus one Evidence per matching file
    (Evidence.location is prefixed with the claim_id - the convention
    evaluate_blockers/compute_empirical_features in novelty_and_publishability.py
    use to link evidence back to a claim). No match produces a non-blocking
    Finding instead of a claim with nothing behind it."""
    claims: list[Claim] = []
    evidence: list[Evidence] = []
    findings: list[Finding] = []

    if not domain_profile:
        return claims, evidence, findings

    for item in domain_profile.get("completeness_checklist", []):
        keywords = [k.lower() for k in item.get("keywords", [])]
        matches = sorted(
            (asset for asset in assets if any(keyword in asset.relative_path.lower() for keyword in keywords)),
            key=lambda asset: asset.relative_path,
        )[:MAX_EVIDENCE_PER_CLAIM]
        if not matches:
            findings.append(Finding(
                finding_id=new_id("FINDING"),
                project_id=project_id,
                severity="MEDIUM",
                rule_id=f"missing_checklist_evidence:{item['id']}",
                message=f"No files matched the '{item['id']}' checklist item: {item.get('description', '')}",
                blocking=False,
            ))
            continue

        claim_id = new_id("CLAIM")
        claim_evidence: list[Evidence] = []
        for asset in matches:
            evidence_type = EXTENSION_EVIDENCE_TYPE.get(
                Path(asset.relative_path).suffix.lower(), "COMPUTATIONAL_RESULT"
            )
            claim_evidence.append(Evidence(
                evidence_id=new_id("EVIDENCE"),
                project_id=project_id,
                source_artifact_id=asset.artifact_id,
                evidence_type=evidence_type,
                location=f"{claim_id}:{asset.relative_path}",
                excerpt_or_value=f"File matched checklist item '{item['id']}' by keyword.",
                extraction_confidence=0.6,  # deterministic keyword match, not a verified read
            ))

        claims.append(Claim(
            claim_id=claim_id,
            project_id=project_id,
            text=item.get("description", item["id"]),
            claim_type="RESULT",
            importance=item.get("importance", "SUPPORTING"),
            evidence_ids=[e.evidence_id for e in claim_evidence],
            status="CANDIDATE",
        ))
        evidence.extend(claim_evidence)

    return claims, evidence, findings


def to_evidence_packet(
    *, project_id: str, domain_profile_id: str, claims: list[Claim], evidence: list[Evidence],
) -> EvidencePacket:
    """Translate the canonical (persisted) Claim/Evidence into
    novelty_and_publishability.py's own CandidateClaim/EvidenceItem shape,
    right before calling its functions. Canonical models stay the system of
    record; this is a call-time adapter only."""
    return EvidencePacket(
        project_id=project_id,
        domain_profile_id=domain_profile_id,
        claims=[
            CandidateClaim(
                claim_id=c.claim_id,
                text=c.text,
                claim_type=_CLAIM_TYPE_MAP.get(c.claim_type, "RESULT"),
                importance=c.importance,
            )
            for c in claims
        ],
        evidence=[
            EvidenceItem(
                evidence_id=e.evidence_id,
                source_artifact_id=e.source_artifact_id,
                evidence_type=e.evidence_type,
                location=e.location,
                excerpt_or_value=e.excerpt_or_value,
                extraction_confidence=e.extraction_confidence,
            )
            for e in evidence
        ],
        empirical_features=[],
        findings=[],
    )
