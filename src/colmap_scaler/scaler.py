"""Apply scale transformation to COLMAP reconstruction."""

import logging
import numpy as np
import pycolmap
from pathlib import Path

logger = logging.getLogger(__name__)


class ReconstructionScaler:
    """Apply scale transformation to COLMAP reconstruction."""

    def __init__(self, reconstruction: pycolmap.Reconstruction) -> None:
        """Initialize reconstruction scaler.

        :param reconstruction: COLMAP reconstruction to scale.
        """
        self.reconstruction = reconstruction

    def apply_scale(self, scale_factor: float) -> None:
        """Apply uniform scale to reconstruction.

        :param scale_factor: Scale factor to apply.
        """
        logger.info(f"\nApplying scale factor: {scale_factor:.6f}")

        # Create similarity transformation with only scale
        # pycolmap uses scale in the form: X_new = scale * X_old
        sim3d = pycolmap.Sim3d()
        sim3d.scale = scale_factor

        # Apply transformation
        self.reconstruction.transform(sim3d)

        logger.info("Scale transformation applied successfully")

    def save_reconstruction(
        self,
        output_path: Path,
        export_text: bool = True,
        export_ply: bool = True
    ) -> None:
        """Save scaled reconstruction to disk.

        :param output_path: Output directory path.
        :param export_text: Whether to export text format.
        :param export_ply: Whether to export PLY point cloud.
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save binary format (default COLMAP format)
        logger.info(f"\nSaving scaled reconstruction to {output_path}")
        self.reconstruction.write(output_path)
        logger.info("Saved binary reconstruction")

        # Save text format
        if export_text:
            text_path = output_path.parent / f"{output_path.name}_text"
            text_path.mkdir(parents=True, exist_ok=True)
            self.reconstruction.write_text(text_path)
            logger.info(f"Saved text reconstruction to {text_path}")

        # Export PLY point cloud
        if export_ply:
            ply_path = output_path.parent / f"{output_path.name}.ply"
            self.reconstruction.export_PLY(str(ply_path))
            logger.info(f"Exported PLY point cloud to {ply_path}")

    def get_reconstruction_stats(self) -> dict:
        """Get statistics about the reconstruction.

        :returns: Dictionary with reconstruction statistics.
        """
        # Calculate bounding box
        points3D = []
        for point3D in self.reconstruction.points3D.values():
            points3D.append(point3D.xyz)

        if points3D:
            points3D = np.array(points3D)
            bbox_min = points3D.min(axis=0)
            bbox_max = points3D.max(axis=0)
            bbox_size = bbox_max - bbox_min
            centroid = points3D.mean(axis=0)
        else:
            bbox_min = bbox_max = bbox_size = centroid = np.zeros(3)

        # Calculate camera positions extent
        camera_positions = []
        for image in self.reconstruction.images.values():
            # Get camera position in world coordinates (projection center)
            camera_positions.append(image.projection_center())

        if camera_positions:
            camera_positions = np.array(camera_positions)
            camera_extent = np.ptp(camera_positions, axis=0)
            camera_centroid = camera_positions.mean(axis=0)
        else:
            camera_extent = camera_centroid = np.zeros(3)

        stats = {
            "num_cameras": len(self.reconstruction.cameras),
            "num_images": len(self.reconstruction.images),
            "num_points3D": len(self.reconstruction.points3D),
            "bbox_min": bbox_min.tolist(),
            "bbox_max": bbox_max.tolist(),
            "bbox_size": bbox_size.tolist(),
            "scene_extent": np.linalg.norm(bbox_size),
            "centroid": centroid.tolist(),
            "camera_extent": camera_extent.tolist(),
            "camera_centroid": camera_centroid.tolist(),
        }

        return stats

    def print_reconstruction_stats(self, stats: dict, title: str = "Reconstruction Statistics") -> None:
        """Print reconstruction statistics.

        :param stats: Statistics dictionary.
        :param title: Title to display.
        """
        logger.info(f"\n{title}:")
        logger.info(f"  Cameras: {stats['num_cameras']}")
        logger.info(f"  Images: {stats['num_images']}")
        logger.info(f"  3D Points: {stats['num_points3D']}")
        logger.info(f"  Scene extent: {stats['scene_extent']:.4f} units")
        logger.info(f"  Bounding box size: [{stats['bbox_size'][0]:.4f}, {stats['bbox_size'][1]:.4f}, {stats['bbox_size'][2]:.4f}]")
        logger.info(f"  Camera extent: [{stats['camera_extent'][0]:.4f}, {stats['camera_extent'][1]:.4f}, {stats['camera_extent'][2]:.4f}]")