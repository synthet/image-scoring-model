"""Wildlife eye localization and heuristic eye-focus scoring."""

from eye_quality.pipeline import score_image
from eye_quality.schemas import EyeQualityPipelineResult

__all__ = ["score_image", "EyeQualityPipelineResult"]
__version__ = "0.1.0"
