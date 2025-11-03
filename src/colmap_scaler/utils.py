"""Utility functions for COLMAP scaler."""

import json
import logging
import random
from pathlib import Path

logger = logging.getLogger(__name__)


def get_image_files(images_dir: Path) -> list[Path]:
    """Get all image files from a directory.

    :param images_dir: Directory containing images.
    :returns: List of image file paths.
    """
    # Common image extensions
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

    image_files = []
    for ext in extensions:
        image_files.extend(images_dir.glob(f"*{ext}"))
        image_files.extend(images_dir.glob(f"*{ext.upper()}"))

    # Sort for consistent ordering
    image_files.sort()

    return image_files


def select_images(
    images_dir: Path, num_images: int | None = None, image_list_file: Path | None = None
) -> list[Path]:
    """Select images to process based on configuration.

    :param images_dir: Directory containing images.
    :param num_images: Number of random images to select (if specified).
    :param image_list_file: File containing specific image names (if specified).
    :returns: List of selected image paths.
    """
    if image_list_file and image_list_file.exists():
        # Read specific image names from file
        logger.info(f"Reading image list from {image_list_file}")

        with open(image_list_file, "r") as f:
            image_names = [line.strip() for line in f if line.strip()]

        # Convert names to paths
        selected = []
        for name in image_names:
            image_path = images_dir / name
            if image_path.exists():
                selected.append(image_path)
            else:
                logger.warning(f"Image not found: {name}")

        logger.info(f"Selected {len(selected)} images from list")
        return selected

    # Get all available images
    all_images = get_image_files(images_dir)

    if not all_images:
        logger.error(f"No images found in {images_dir}")
        return []

    if num_images and num_images < len(all_images):
        # Random selection
        logger.info(f"Randomly selecting {num_images} from {len(all_images)} images")
        selected = random.sample(all_images, num_images)
        selected.sort()  # Sort for consistent ordering
        return selected

    # Process all images
    logger.info(f"Processing all {len(all_images)} images")
    return all_images


def save_json(data: dict, output_path: Path, indent: int = 2) -> None:
    """Save data to JSON file.

    :param data: Data to save.
    :param output_path: Output file path.
    :param indent: JSON indentation level.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(data, f, indent=indent)

    logger.info(f"Saved results to {output_path}")


def load_json(file_path: Path) -> dict:
    """Load data from JSON file.

    :param file_path: Path to JSON file.
    :returns: Loaded data dictionary.
    """
    with open(file_path, "r") as f:
        return json.load(f)


def print_detection_summary(summary: dict) -> None:
    """Print detection summary to console.

    :param summary: Detection summary dictionary.
    """
    logger.info("\nDetection Summary:")
    logger.info(f"  Total images processed: {summary['total_images']}")
    logger.info(f"  Images with markers: {summary['images_with_markers']}")
    logger.info(f"  Unique markers detected: {summary['unique_markers']}")
    logger.info(f"  Average markers per image: {summary['avg_markers_per_image']}")

    if summary["unique_markers"]:
        logger.info(f"  Marker IDs: {', '.join(map(str, summary['unique_markers']))}")

    detection_rate = (
        summary["images_with_markers"] / summary["total_images"] * 100
        if summary["total_images"] > 0
        else 0
    )
    logger.info(f"  Detection rate: {detection_rate:.1f}%")


def validate_paths(config) -> bool:
    """Validate required paths exist.

    :param config: Detection configuration.
    :returns: True if all paths are valid.
    """
    valid = True

    if not config.images_path.exists():
        logger.error(f"Images directory not found: {config.images_path}")
        valid = False

    if config.image_list_file and not config.image_list_file.exists():
        logger.error(f"Image list file not found: {config.image_list_file}")
        valid = False

    return valid
