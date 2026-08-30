"""WMEval Lite: strict case/candidate loading plus the unified suite runner."""

from .candidate_discovery import (
    CandidateRow,
    discover_candidate_directory,
    load_candidate_manifest,
)
from .case_loader import ALLOWED_CASES, SCENE_MODULES, LiteCase, discover_cases, load_case
from .suite import run_suite
from .suite_loader import DimensionSpec, Suite, SuiteValidationError, load_suite
from .validation import (
    CandidateValidationError,
    CaseValidationError,
    validate_candidate_video,
    validate_case_bundle,
)

__all__ = [
    "ALLOWED_CASES",
    "SCENE_MODULES",
    "CandidateRow",
    "CandidateValidationError",
    "CaseValidationError",
    "DimensionSpec",
    "LiteCase",
    "Suite",
    "SuiteValidationError",
    "discover_candidate_directory",
    "discover_cases",
    "load_candidate_manifest",
    "load_case",
    "load_suite",
    "run_suite",
    "validate_candidate_video",
    "validate_case_bundle",
]
