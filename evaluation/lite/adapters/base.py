"""Shared adapter contract: one evaluator call per (dimension, case, candidate)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from ..case_loader import LiteCase
from ..candidate_discovery import CandidateRow
from ..suite_loader import DimensionSpec, Suite

# Row/result statuses shared across the suite layer.
STATUS_COMPLETED = "completed"
STATUS_DRY_RUN = "dry_run"
STATUS_MISSING_CANDIDATE = "missing_candidate"
STATUS_INVALID_CANDIDATE = "invalid_candidate"
STATUS_MISSING_RUBRIC = "missing_rubric"
STATUS_SKIPPED_NO_API_KEY = "skipped_no_api_key"
STATUS_UNSUPPORTED = "unsupported"
STATUS_ERROR = "error"


@dataclass
class CaseResult:
    status: str
    metrics: dict[str, Any] = field(default_factory=dict)
    full_report: dict[str, Any] | None = None
    error: str | None = None
    # Candidate-vs-reference video properties, recorded for every scored row.
    input_contract: dict[str, Any] | None = None


class DimensionEvaluator(Protocol):
    def evaluate(
        self,
        *,
        suite: Suite,
        dimension: DimensionSpec,
        case: LiteCase,
        candidate: CandidateRow,
        candidate_video: Path,
        device: str | None = None,
    ) -> CaseResult: ...
