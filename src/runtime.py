"""Reuse one model session in the sequential Gradio analysis queue."""

from functools import lru_cache
from pathlib import Path

from src.detector import LaneDetector


@lru_cache(maxsize=1)
def _load_detector(path: str, modified_ns: int, size: int, device: str, crop: float | None):
    return LaneDetector(path, device=device, road_crop_ratio=crop)


def get_detector(model_path: str, device: str = 'auto', road_crop_ratio: float | None = None):
    path = Path(model_path).resolve()
    stat = path.stat()
    return _load_detector(str(path), stat.st_mtime_ns, stat.st_size, device, road_crop_ratio)
