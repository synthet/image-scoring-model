"""Package init for localization."""

from eye_quality.localization.bird_detector import BirdDetector
from eye_quality.localization.pose_model import PoseLocalizer

__all__ = ["BirdDetector", "PoseLocalizer"]
