"""Video frame extractor.

Extracts representative frames from drone video footage for downstream
detection. Supports time-based or scene-change based sampling.

Implementation lands in MVP-1.
"""

from __future__ import annotations

from pathlib import Path


def extract_frames(
    video_path: str | Path,
    output_dir: str | Path,
    *,
    fps: float | None = None,
    every_n_frames: int | None = None,
) -> list[Path]:
    """Extract frames from a video file.

    Exactly one of ``fps`` or ``every_n_frames`` must be supplied.

    Args:
        video_path: Source video file.
        output_dir: Directory to write extracted frames into.
        fps: Frames per second to sample.
        every_n_frames: Take every Nth frame.

    Returns:
        List of paths to the extracted frame images.

    Raises:
        NotImplementedError: Until MVP-1 implementation lands.
    """
    raise NotImplementedError("Video frame extractor implementation lands in MVP-1.")
