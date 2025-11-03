"""Configuration classes for COLMAP scaler."""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class MarkerType(Enum):
    """Supported marker types."""
    ARUCO = "aruco"
    APRILTAG = "apriltag"


class ArucoDictionary(Enum):
    """OpenCV ArUco dictionary types."""
    DICT_4X4_50 = "DICT_4X4_50"
    DICT_4X4_100 = "DICT_4X4_100"
    DICT_4X4_250 = "DICT_4X4_250"
    DICT_4X4_1000 = "DICT_4X4_1000"
    DICT_5X5_50 = "DICT_5X5_50"
    DICT_5X5_100 = "DICT_5X5_100"
    DICT_5X5_250 = "DICT_5X5_250"
    DICT_5X5_1000 = "DICT_5X5_1000"
    DICT_6X6_50 = "DICT_6X6_50"
    DICT_6X6_100 = "DICT_6X6_100"
    DICT_6X6_250 = "DICT_6X6_250"
    DICT_6X6_1000 = "DICT_6X6_1000"
    DICT_7X7_50 = "DICT_7X7_50"
    DICT_7X7_100 = "DICT_7X7_100"
    DICT_7X7_250 = "DICT_7X7_250"
    DICT_7X7_1000 = "DICT_7X7_1000"
    DICT_ARUCO_ORIGINAL = "DICT_ARUCO_ORIGINAL"
    DICT_APRILTAG_16h5 = "DICT_APRILTAG_16h5"
    DICT_APRILTAG_25h9 = "DICT_APRILTAG_25h9"
    DICT_APRILTAG_36h10 = "DICT_APRILTAG_36h10"
    DICT_APRILTAG_36h11 = "DICT_APRILTAG_36h11"


@dataclass
class DetectionConfig:
    """Configuration for marker detection."""

    # Input/output paths
    images_path: Path
    output_path: Path

    # Marker settings
    marker_type: MarkerType = MarkerType.ARUCO
    marker_size: float = 0.0
    dictionary: str | None = None

    # Image selection
    num_images: int | None = None
    image_list_file: Path | None = None

    # Detection parameters
    refine_corners: bool = True
    min_marker_perimeter: int = 50
    max_marker_perimeter: int = 10000

    # Visualization
    visualize: bool = False

    def __post_init__(self) -> None:
        """Validate and setup configuration."""
        # Convert paths to Path objects
        self.images_path = Path(self.images_path)
        self.output_path = Path(self.output_path)

        if self.image_list_file:
            self.image_list_file = Path(self.image_list_file)

        # Create output directory for JSON
        self.output_path.parent.mkdir(parents=True, exist_ok=True)


@dataclass
class DetectionResult:
    """Result of marker detection for a single image."""

    image_name: str
    markers: list = field(default_factory=list)

    def add_marker(self, marker_id: int, corners: list) -> None:
        """Add a detected marker.
        
        :param marker_id: ID of the marker.
        :param corners: List of corner coordinates.
        """
        self.markers.append({
            "id": marker_id,
            "corners": corners
        })

    def has_markers(self) -> bool:
        """Check if any markers were detected.
        
        :returns: True if markers were detected.
        """
        return len(self.markers) > 0

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization.
        
        :returns: Dictionary representation.
        """
        return {
            "markers": self.markers
        }


@dataclass
class DetectionSummary:
    """Summary of all detections."""

    metadata: dict = field(default_factory=dict)
    detections: dict = field(default_factory=dict)

    def add_detection(self, image_name: str, result: DetectionResult) -> None:
        """Add a detection result.
        
        :param image_name: Name of the image.
        :param result: Detection result for the image.
        """
        if result.has_markers():
            self.detections[image_name] = result.to_dict()

    def compute_summary(self) -> dict:
        """Compute summary statistics.
        
        :returns: Summary statistics dictionary.
        """
        total_images = len(self.detections)
        images_with_markers = sum(1 for det in self.detections.values()
                                 if det.get("markers"))

        # Collect unique marker IDs
        unique_markers = set()
        for det in self.detections.values():
            for marker in det.get("markers", []):
                unique_markers.add(marker["id"])

        # Calculate average markers per image
        total_markers = sum(len(det.get("markers", []))
                           for det in self.detections.values())
        avg_markers = total_markers / total_images if total_images > 0 else 0

        return {
            "total_images": total_images,
            "images_with_markers": images_with_markers,
            "unique_markers": sorted(list(unique_markers)),
            "avg_markers_per_image": round(avg_markers, 2)
        }

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization.
        
        :returns: Dictionary representation.
        """
        return {
            "metadata": self.metadata,
            "detections": self.detections,
            "summary": self.compute_summary()
        }