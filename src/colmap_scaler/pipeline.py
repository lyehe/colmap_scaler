"""Main pipeline for COLMAP reconstruction scaling."""

import json
import logging
import sys
from pathlib import Path
from datetime import datetime
import pycolmap

from .config import DetectionConfig, DetectionSummary
from .detector import MarkerDetector
from .triangulator import CornerTriangulator
from .scale_estimator import ScaleEstimator
from .scaler import ReconstructionScaler
from .utils import select_images, save_json, print_detection_summary, validate_paths

logger = logging.getLogger(__name__)


class ScalingPipeline:
    """Orchestrates the complete marker detection and reconstruction scaling pipeline."""

    def __init__(self, config) -> None:
        """Initialize pipeline with configuration.

        :param config: Configuration object with all pipeline parameters.
        """
        self.config = config
        self.dataset_folder = None
        self.images_folder = None
        self.input_sparse_path = None
        self.sparse_path = None
        self.backup_path = None
        self.log_path = None
        self.log_lines = []

    def setup_paths(self) -> None:
        """Setup and validate all required paths."""
        # Set defaults
        output_name = self.config.output_name if self.config.output_name else "0"
        backup_name = self.config.backup_name if self.config.backup_name else f"{output_name}_before_scale"

        # Handle input path
        input_path = Path(self.config.input)
        if input_path.name.lower() == "images":
            self.dataset_folder = input_path.parent
            self.images_folder = input_path
            logger.info(f"Detected /images path, using dataset folder: {self.dataset_folder}")
        else:
            self.dataset_folder = input_path
            self.images_folder = input_path / "images"
            if not self.images_folder.exists():
                logger.error(f"Error: Images folder not found at {self.images_folder}")
                logger.info("Expected structure: dataset_folder/images/")
                sys.exit(1)

        # Handle reconstruction input path (where to load from)
        recon_path = self.dataset_folder if self.config.reconstruction is None else Path(self.config.reconstruction)

        if (recon_path / "sparse").exists():
            sparse_folder = recon_path / "sparse"
            input_sparse_path = sparse_folder / "0"
        elif recon_path.name in ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9") or recon_path.parent.name == "sparse":
            input_sparse_path = recon_path
            sparse_folder = recon_path.parent
        elif recon_path.name == "sparse":
            sparse_folder = recon_path
            input_sparse_path = sparse_folder / "0"
        else:
            logger.error(f"Error: Could not identify reconstruction structure at {recon_path}")
            sys.exit(1)

        if not input_sparse_path.exists():
            logger.error(f"Error: Reconstruction not found at {input_sparse_path}")
            sys.exit(1)

        # Output paths (where to save)
        self.sparse_path = sparse_folder / output_name
        self.backup_path = sparse_folder / backup_name
        self.log_path = self.dataset_folder / "scale.log"

        # Store input path for loading
        self.input_sparse_path = input_sparse_path

        logger.info(f"Dataset folder: {self.dataset_folder}")
        logger.info(f"Images folder: {self.images_folder}")
        logger.info(f"Input reconstruction: {self.input_sparse_path}")
        logger.info(f"Output reconstruction: {self.sparse_path}")
        logger.info(f"Log output: {self.log_path}\n")

    def run_detection(self) -> tuple[dict, dict, dict]:
        """Run marker detection phase.

        :returns: Tuple of (detection_summary dict, detections dict, metadata dict).
        """
        logger.info("="*60)
        logger.info("PHASE 1: Marker Detection")
        logger.info("="*60 + "\n")

        self.log_lines.append("-"*60)
        self.log_lines.append("PHASE 1: MARKER DETECTION")
        self.log_lines.append("-"*60)

        # Infer marker type from dictionary (for internal use only)
        from .config import MarkerType
        if self.config.dictionary and "APRILTAG" in self.config.dictionary.upper():
            marker_type = MarkerType.APRILTAG
        else:
            # Default to ARUCO (covers ArUco dicts and auto-detection)
            marker_type = MarkerType.ARUCO

        # Create detection config
        detections_path = self.dataset_folder / "detections.json"
        detection_config = DetectionConfig(
            images_path=self.images_folder,
            output_path=detections_path,
            marker_type=marker_type,
            marker_size=self.config.marker_size,
            dictionary=self.config.dictionary,
            num_images=self.config.num_images,
            image_list_file=self.config.image_list,
            visualize=self.config.visualize,
            refine_corners=self.config.refine_corners,
            min_marker_perimeter=self.config.min_perimeter,
            max_marker_perimeter=self.config.max_perimeter
        )

        # Validate and select images
        if not validate_paths(detection_config):
            sys.exit(1)

        image_paths = select_images(
            detection_config.images_path,
            detection_config.num_images,
            detection_config.image_list_file
        )

        if not image_paths:
            logger.error("No images to process!")
            sys.exit(1)

        # Initialize detector and process images
        logger.info(f"Marker type: {detection_config.marker_type.value}")
        logger.info(f"Marker size: {detection_config.marker_size * 1000:.1f}mm ({detection_config.marker_size:.3f}m)")
        detector = MarkerDetector(detection_config)

        self.log_lines.append(f"Marker type: {detection_config.marker_type.value}")
        self.log_lines.append(f"Marker size: {detection_config.marker_size * 1000:.1f}mm ({detection_config.marker_size:.3f}m)")
        self.log_lines.append(f"Dictionary: {detector.dictionary_name or 'auto-detected'}")

        logger.info(f"\nProcessing {len(image_paths)} images...")
        results = detector.process_image_list(image_paths)

        # Create and save summary
        summary = DetectionSummary()
        summary.metadata = {
            "marker_type": detection_config.marker_type.value,
            "marker_size_meters": detection_config.marker_size,
            "marker_size_mm": detection_config.marker_size * 1000,
            "dictionary": detector.dictionary_name or "auto-detected",
            "timestamp": datetime.now().isoformat(),
            "images_path": str(detection_config.images_path),
            "num_images_processed": len(image_paths)
        }

        for image_name, result in results.items():
            summary.add_detection(image_name, result)

        detection_summary = summary.compute_summary()
        print_detection_summary(detection_summary)

        # Log detection results
        detection_rate = (detection_summary['images_with_markers'] / detection_summary['total_images'] * 100
                         if detection_summary['total_images'] > 0 else 0)

        self.log_lines.append(f"Total images processed: {detection_summary['total_images']}")
        self.log_lines.append(f"Images with markers: {detection_summary['images_with_markers']}")
        self.log_lines.append(f"Detection rate: {detection_rate:.1f}%")
        self.log_lines.append(f"Unique markers: {detection_summary['unique_markers']}")
        self.log_lines.append(f"Average markers per image: {detection_summary['avg_markers_per_image']:.2f}")

        # Save JSON results (optional)
        if self.config.export_json:
            save_json(summary.to_dict(), detection_config.output_path)
            logger.info(f"\nDetection results saved to {detection_config.output_path}")

        if detection_config.visualize:
            tags_dir = detection_config.images_path.parent / "tags"
            logger.info(f"Visualizations saved to {tags_dir}")
            self.log_lines.append(f"Visualizations: {tags_dir}")

        self.log_lines.append("")

        # Return data for next phase
        detections_dict = {}
        for image_name, result in results.items():
            if result.has_markers():
                detections_dict[image_name] = result.to_dict()

        return detection_summary, detections_dict, summary.metadata

    def run_scaling(self, detections: dict, metadata: dict) -> dict:
        """Run reconstruction scaling phase.

        :param detections: Dictionary of detections from detection phase.
        :param metadata: Metadata dictionary from detection phase.
        :returns: Scaling report containing all results.
        """
        logger.info("\n" + "="*60)
        logger.info("PHASE 2: Reconstruction Scaling")
        logger.info("="*60 + "\n")

        self.log_lines.append("-"*60)
        self.log_lines.append("PHASE 2: RECONSTRUCTION SCALING")
        self.log_lines.append("-"*60)

        # Use detections from previous phase
        marker_size = metadata.get("marker_size_meters", self.config.marker_size)

        logger.info(f"Processing {len(detections)} images with detections\n")

        # Load reconstruction
        logger.info("Loading COLMAP reconstruction...")
        reconstruction = pycolmap.Reconstruction(str(self.input_sparse_path))

        # Get initial stats
        scaler = ReconstructionScaler(reconstruction)
        stats_before = scaler.get_reconstruction_stats()
        scaler.print_reconstruction_stats(stats_before, "Original Reconstruction")

        self._log_reconstruction_stats(stats_before, "Original Reconstruction")

        # Triangulate markers
        validated = self._triangulate_markers(reconstruction, detections)

        # Estimate scale
        scale_factor, scale_stats = self._estimate_scale(marker_size, validated)

        # Apply scaling
        scaler.apply_scale(scale_factor)
        stats_after = scaler.get_reconstruction_stats()
        scaler.print_reconstruction_stats(stats_after, "Scaled Reconstruction")

        self._log_reconstruction_stats(stats_after, "Scaled Reconstruction", is_scaled=True)

        reduction = stats_before['scene_extent'] / stats_after['scene_extent']
        self.log_lines.append("")
        self.log_lines.append(f"Scale reduction factor: {reduction:.1f}x")
        self.log_lines.append(f"Before: {stats_before['scene_extent']:.2f} units -> After: {stats_after['scene_extent']:.2f} meters")

        # Backup and save
        self._backup_reconstruction()
        self._save_reconstruction(scaler)

        return {
            "scale_factor": scale_factor,
            "marker_size_meters": marker_size,
            "scale_statistics": scale_stats,
            "reconstruction_before": stats_before,
            "reconstruction_after": stats_after,
        }

    def _triangulate_markers(self, reconstruction: pycolmap.Reconstruction, detections: dict) -> dict[int, dict]:
        """Triangulate marker corners and validate.
        
        :param reconstruction: COLMAP reconstruction.
        :param detections: Dictionary of detections.
        :returns: Validated triangulated markers.
        """
        logger.info("\nTriangulating marker corners...")
        triangulator = CornerTriangulator(reconstruction)

        triangulated = triangulator.triangulate_marker_corners(
            detections,
            min_num_views=self.config.min_num_views
        )

        validated = triangulator.validate_triangulated_corners(
            triangulated,
            max_reprojection_error=self.config.max_reprojection_error
        )

        logger.info(f"Successfully triangulated {len(validated)}/{len(triangulated)} markers")
        self.log_lines.append(f"Triangulated markers: {len(validated)}/{len(triangulated)}")

        if not validated:
            logger.error("Error: No valid markers for scale estimation")
            self.log_lines.append("ERROR: No valid markers for scale estimation")
            self._save_log()
            sys.exit(1)

        return validated

    def _estimate_scale(self, marker_size: float, validated_markers: dict[int, dict]) -> tuple[float, dict]:
        """Estimate scale factor from validated markers.
        
        :param marker_size: Physical marker size in meters.
        :param validated_markers: Dictionary of validated markers.
        :returns: Tuple of (scale factor, statistics dict).
        """
        logger.info("\nEstimating scale factor...")
        estimator = ScaleEstimator(marker_size)

        scale_factor, scale_stats = estimator.estimate_scale(
            validated_markers,
            method="robust_mean"
        )

        self.log_lines.append("Scale estimation method: robust_mean")
        self.log_lines.append(f"Markers used: {list(scale_stats['marker_scales'].keys())}")
        self.log_lines.append(f"Scale factor: {scale_factor:.6f}")
        self.log_lines.append("Scale statistics:")
        self.log_lines.append(f"  Mean: {scale_stats['mean']:.6f}")
        self.log_lines.append(f"  Median: {scale_stats['median']:.6f}")
        self.log_lines.append(f"  Std: {scale_stats['std']:.6f}")
        self.log_lines.append(f"  Std/Mean ratio: {scale_stats['std']/scale_stats['mean']:.4f}")
        self.log_lines.append(f"  Range: [{scale_stats['min']:.6f}, {scale_stats['max']:.6f}]")

        if not estimator.validate_scale(scale_factor, scale_stats):
            logger.warning("Warning: Scale validation failed, but continuing...")
            self.log_lines.append("WARNING: Scale validation failed")

        return scale_factor, scale_stats

    def _log_reconstruction_stats(self, stats: dict, title: str, is_scaled: bool = False) -> None:
        """Log reconstruction statistics.
        
        :param stats: Statistics dictionary.
        :param title: Title for the log section.
        :param is_scaled: Whether the reconstruction is scaled.
        """
        unit = "meters" if is_scaled else "units"

        self.log_lines.append(f"{title}:")
        self.log_lines.append(f"  Cameras: {stats['num_cameras']}")
        self.log_lines.append(f"  Images: {stats['num_images']}")
        self.log_lines.append(f"  3D Points: {stats['num_points3D']}")
        self.log_lines.append(f"  Scene extent: {stats['scene_extent']:.4f} {unit}")
        self.log_lines.append(f"  Bounding box: [{stats['bbox_size'][0]:.4f}, {stats['bbox_size'][1]:.4f}, {stats['bbox_size'][2]:.4f}]")

    def _backup_reconstruction(self) -> None:
        """Create backup of output location if it exists."""
        if not self.sparse_path.exists():
            logger.info(f"\nOutput location {self.sparse_path} does not exist, no backup needed")
            self.log_lines.append("Backup: Not needed (new output location)")
            return

        logger.info("\nCreating backup of existing output reconstruction...")

        if self.backup_path.exists():
            logger.warning(f"Warning: Backup already exists at {self.backup_path}")
            logger.warning("Skipping backup (original backup preserved)")
            self.log_lines.append(f"Backup: Already exists at {self.backup_path}")
        else:
            import shutil
            shutil.copytree(self.sparse_path, self.backup_path)
            logger.info(f"Backup saved to {self.backup_path}")
            self.log_lines.append(f"Backup: Created at {self.backup_path}")

    def _save_reconstruction(self, scaler: ReconstructionScaler) -> None:
        """Save scaled reconstruction.
        
        :param scaler: Reconstruction scaler instance.
        """
        logger.info(f"\nSaving scaled reconstruction to {self.sparse_path}")
        scaler.reconstruction.write(self.sparse_path)
        logger.info("Saved scaled reconstruction (binary format)")

    def _save_log(self) -> None:
        """Save log file."""
        with open(self.log_path, 'w') as f:
            f.write('\n'.join(self.log_lines))

    def save_report(self, detection_summary: dict, scale_report: dict) -> None:
        """Save final JSON report and log file.

        :param detection_summary: Detection phase summary.
        :param scale_report: Scaling phase report.
        """
        # Save JSON report (optional)
        report_path = self.dataset_folder / "scale_report.json"
        if self.config.export_json:
            full_report = {
                **scale_report,
                "detection_summary": detection_summary,
                "backup_location": str(self.backup_path),
                "parameters": {
                    "min_num_views": self.config.min_num_views,
                    "max_reprojection_error": self.config.max_reprojection_error,
                    "scale_method": "robust_mean",
                }
            }

            with open(report_path, 'w') as f:
                json.dump(full_report, f, indent=2)

            logger.info(f"Scale report saved to {report_path}")

        # Save text log
        self.log_lines.append("")
        self.log_lines.append("-"*60)
        self.log_lines.append("OUTPUT FILES")
        self.log_lines.append("-"*60)
        self.log_lines.append(f"Log file: {self.log_path}")
        if self.config.export_json:
            self.log_lines.append(f"Detections: {self.dataset_folder / 'detections.json'}")
            self.log_lines.append(f"Scale report: {report_path}")
        self.log_lines.append(f"Scaled reconstruction: {self.sparse_path}")
        self.log_lines.append(f"Backup reconstruction: {self.backup_path}")
        self.log_lines.append("")
        self.log_lines.append("="*60)
        self.log_lines.append("Pipeline completed successfully!")
        self.log_lines.append("="*60)

        self._save_log()
        logger.info(f"Complete log saved to {self.log_path}")

    def run(self) -> None:
        """Execute complete pipeline."""
        logger.info("="*60)
        logger.info("COLMAP Scaler")
        logger.info("="*60 + "\n")

        # Validate marker size
        if self.config.marker_size <= 0:
            logger.error("Error: Marker size must be positive (in meters)")
            logger.info("Example: --marker-size 0.05 for 50mm markers")
            sys.exit(1)

        # Initialize log
        self.log_lines = []
        self.log_lines.append("="*60)
        self.log_lines.append("COLMAP Scaler - Scale Log Report")
        self.log_lines.append("="*60)
        self.log_lines.append(f"Timestamp: {datetime.now().isoformat()}")

        # Setup paths
        self.setup_paths()
        self.log_lines.append(f"Dataset: {self.dataset_folder}")
        self.log_lines.append("")

        # Run pipeline phases
        detection_summary, detections, metadata = self.run_detection()
        scale_report = self.run_scaling(detections, metadata)

        # Save reports
        self.save_report(detection_summary, scale_report)

        # Print completion message
        logger.info("\n" + "="*60)
        logger.info("Pipeline Complete!")
        logger.info("="*60)
        logger.info(f"Original reconstruction backed up to: {self.backup_path}")
        logger.info(f"Scaled reconstruction saved to: {self.sparse_path}")
        logger.info(f"Complete log saved to: {self.log_path}")
