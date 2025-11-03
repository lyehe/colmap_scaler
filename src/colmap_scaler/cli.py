"""Command-line interface for COLMAP scaler."""

import logging
from dataclasses import dataclass
from pathlib import Path

import tyro

from .pipeline import ScalingPipeline

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(message)s")


@dataclass
class ScalerArgs:
    """COLMAP Scaler - Rescale COLMAP reconstructions using fiducial markers.

    This tool detects ArUco or AprilTag markers in images and uses them
    to rescale COLMAP reconstructions to real-world units.
    """

    input: Path
    """Path to dataset folder or images directory. If path ends with /images, parent folder is used."""

    marker_size: float
    """Physical marker size in meters (e.g., 0.05 for 50mm)."""

    reconstruction: Path | None = None
    """Path to COLMAP reconstruction. Default: dataset_folder/sparse/0"""

    dictionary: str | None = None
    """Marker dictionary (e.g., DICT_4X4_50, DICT_APRILTAG_25h9). Auto-detect if not specified."""

    num_images: int | None = None
    """Process only N randomly selected images. Process all if not specified."""

    image_list: Path | None = None
    """Text file with specific image names to process (one per line)."""

    visualize: bool = False
    """Save visualization images with detected markers in dataset_folder/tags/."""

    refine_corners: bool = True
    """Apply subpixel corner refinement for better accuracy."""

    min_perimeter: int = 50
    """Minimum marker perimeter in pixels."""

    max_perimeter: int = 10000
    """Maximum marker perimeter in pixels."""

    output_name: str | None = None
    """Output folder name under sparse/. Default: 0"""

    backup_name: str | None = None
    """Backup folder name. Default: {output_name}_before_scale"""

    min_num_views: int = 2
    """Minimum number of views required to triangulate a corner."""

    max_reprojection_error: float = 2.0
    """Maximum reprojection error in pixels."""

    export_json: bool = False
    """Export scale_report.json and detections.json files."""


def main() -> None:
    """Main entry point for COLMAP Scaler."""
    args = tyro.cli(ScalerArgs)

    # Create and run pipeline
    pipeline = ScalingPipeline(args)
    pipeline.run()


if __name__ == "__main__":
    main()
