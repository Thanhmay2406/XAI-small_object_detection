"""XAI feature extraction and saliency helpers."""

from .cam import build_cam_extractor, preprocess_image_for_yolo
from .case_selection import CuratedCase, build_curated_cases
from .evidence_metrics import compute_evidence_metrics
from .evidence_pipeline import run_xai_evidence_extraction
from .evidence_review import review_xai_evidence
from .intervention_design import design_phase8_interventions
from .manual_review import prepare_manual_evidence_review, summarize_manual_review

__all__ = [
    "CuratedCase",
    "build_cam_extractor",
    "build_curated_cases",
    "compute_evidence_metrics",
    "design_phase8_interventions",
    "prepare_manual_evidence_review",
    "preprocess_image_for_yolo",
    "run_xai_evidence_extraction",
    "review_xai_evidence",
    "summarize_manual_review",
]
