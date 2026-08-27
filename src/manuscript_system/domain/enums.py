from __future__ import annotations

from typing import Literal

RunStatus = Literal[
    "CREATED",
    "RUNNING",
    "WAITING_FOR_HUMAN",
    "SUCCEEDED",
    "FAILED",
    "BLOCKED",
    "CANCELLED",
]

ReadinessStatus = Literal[
    "READY_FOR_MANUSCRIPT",
    "DRAFTABLE_WITH_WARNINGS",
    "NEEDS_ADDITIONAL_ANALYSIS",
    "NEEDS_RESEARCH_COMPLETION",
    "INSUFFICIENT_EVIDENCE",
    "BLOCKED",
]

AssessmentDecision = Literal[
    "APPROVE_MANUSCRIPT_PLANNING",
    "APPROVE_COMPLETION_PLAN",
    "REQUEST_REASSESSMENT",
    "BLOCK_RUN",
]

ClaimType = Literal[
    "FACT", "RESULT", "INTERPRETATION", "HYPOTHESIS", "LITERATURE_CLAIM", "SUGGESTION"
]

ClaimImportance = Literal["CENTRAL", "SUPPORTING", "CONTEXTUAL"]

EvidenceType = Literal[
    "USER_INPUT", "EXPERIMENTAL_RESULT", "COMPUTATIONAL_RESULT", "LITERATURE",
    "FIGURE", "TABLE", "CODE", "INFERENCE",
]

FindingSeverity = Literal["INFO", "LOW", "MEDIUM", "HIGH", "CRITICAL"]

# VERIFIED = found a real, matching record via literature search. UNVERIFIED =
# resolved in the bibliography but no matching record found - report only,
# never treated as proof of fabrication. BROKEN_KEY = \cite{} used a key with
# no matching bibliography entry at all - an unambiguous authoring error.
# NON_LITERATURE = the bibliography entry isn't a paper (software, a dataset
# record, a standard) - no literature database will ever "verify" it, so it's
# never searched at all rather than reported as a false UNVERIFIED.
CitationVerificationStatus = Literal["VERIFIED", "UNVERIFIED", "BROKEN_KEY", "NON_LITERATURE"]

# SUPPORTED_BY_DATA = a value in the linked data project matches, at the same
# decimal precision the manuscript reported it, a number stated in the
# manuscript. NOT_FOUND_IN_DATA = no such match - report only, not evidence
# of fabrication: plenty of legitimate reported numbers are derived/summary
# values never saved verbatim anywhere.
NumericCrossCheckStatus = Literal["SUPPORTED_BY_DATA", "NOT_FOUND_IN_DATA"]

# MISSING_DATA = a completeness_checklist item had no matching file at all
# (evidence_extraction.py never even produced a claim for it). VALIDATION =
# a claim exists and has evidence, but an assessor flagged something missing
# about it (e.g. no quantitative content inspected, no benchmark comparison).
CompletionTaskCategory = Literal["MISSING_DATA", "VALIDATION"]
CompletionTaskPriority = Literal["REQUIRED", "RECOMMENDED"]
