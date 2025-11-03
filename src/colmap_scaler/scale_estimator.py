"""Estimate scale factor from triangulated marker corners."""

import logging
import numpy as np

logger = logging.getLogger(__name__)


class ScaleEstimator:
    """Estimate scale factor from markers with known size."""

    def __init__(self, marker_size_meters: float) -> None:
        """Initialize scale estimator.

        :param marker_size_meters: Physical marker size in meters.
        """
        self.marker_size = marker_size_meters

    def estimate_scale(
        self,
        triangulated_markers: dict[int, dict],
        method: str = "robust_mean"
    ) -> tuple[float, dict]:
        """Estimate scale factor from triangulated markers.

        :param triangulated_markers: Dictionary of triangulated marker data.
        :param method: Method to use ("simple", "robust_mean", "median").
        :returns: Tuple of (scale factor, statistics dict).
        """
        if not triangulated_markers:
            raise ValueError("No valid markers to estimate scale from")

        # Calculate scale for each marker
        marker_scales = {}
        for marker_id, data in triangulated_markers.items():
            if not data.get("valid", False):
                continue

            corners_3d = np.array(data["corners_3d"])
            scale = self._estimate_scale_from_corners(corners_3d)

            if scale is not None:
                marker_scales[marker_id] = scale

        if not marker_scales:
            raise ValueError("Failed to estimate scale from any marker")

        logger.info("\nScale estimates from individual markers:")
        for marker_id, scale in marker_scales.items():
            logger.info(f"  Marker {marker_id}: {scale:.4f}")

        # Combine scales using specified method
        scales = list(marker_scales.values())
        final_scale = self._combine_scales(scales, method)

        # Calculate statistics
        stats = {
            "scale_factor": final_scale,
            "num_markers": len(marker_scales),
            "marker_scales": marker_scales,
            "mean": np.mean(scales),
            "median": np.median(scales),
            "std": np.std(scales),
            "min": np.min(scales),
            "max": np.max(scales),
            "method": method,
        }

        return final_scale, stats

    def _estimate_scale_from_corners(self, corners_3d: np.ndarray) -> float | None:
        """Estimate scale from a single marker's 3D corners.

        :param corners_3d: 4x3 array of corner positions.
        :returns: Scale factor or None if estimation failed.
        """
        if corners_3d.shape != (4, 3):
            return None

        # Calculate edge lengths in 3D
        edges = []
        edge_pairs = [
            (0, 1),  # Top edge
            (1, 2),  # Right edge
            (2, 3),  # Bottom edge
            (3, 0),  # Left edge
        ]

        for i, j in edge_pairs:
            length_3d = np.linalg.norm(corners_3d[i] - corners_3d[j])
            edges.append(length_3d)

        # Calculate diagonal lengths
        diag1 = np.linalg.norm(corners_3d[0] - corners_3d[2])
        diag2 = np.linalg.norm(corners_3d[1] - corners_3d[3])

        # Expected lengths
        edge_expected = self.marker_size
        diag_expected = self.marker_size * np.sqrt(2)

        # Calculate scale for each measurement
        scales = []
        for edge in edges:
            if edge > 0:
                scales.append(edge_expected / edge)

        if diag1 > 0:
            scales.append(diag_expected / diag1)
        if diag2 > 0:
            scales.append(diag_expected / diag2)

        if not scales:
            return None

        # Return median scale (robust to outliers)
        return np.median(scales)

    def _combine_scales(self, scales: list[float], method: str) -> float:
        """Combine multiple scale estimates.

        :param scales: List of scale estimates.
        :param method: Combination method.
        :returns: Combined scale factor.
        """
        if method == "simple":
            return np.mean(scales)
        elif method == "median":
            return np.median(scales)
        elif method == "robust_mean":
            # Remove outliers using IQR method
            q1, q3 = np.percentile(scales, [25, 75])
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            filtered = [s for s in scales if lower <= s <= upper]
            return np.mean(filtered) if filtered else np.mean(scales)
        else:
            raise ValueError(f"Unknown method: {method}")

    def validate_scale(
        self,
        scale_factor: float,
        stats: dict,
        max_std_ratio: float = 0.1
    ) -> bool:
        """Validate estimated scale factor.

        :param scale_factor: Estimated scale factor.
        :param stats: Scale statistics.
        :param max_std_ratio: Maximum allowed std/mean ratio.
        :returns: True if scale is valid.
        """
        std_ratio = stats["std"] / stats["mean"]

        logger.info("\nScale Validation:")
        logger.info(f"  Scale factor: {scale_factor:.4f}")
        logger.info(f"  Mean: {stats['mean']:.4f}")
        logger.info(f"  Std: {stats['std']:.4f}")
        logger.info(f"  Std/Mean ratio: {std_ratio:.4f}")
        logger.info(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")

        if std_ratio > max_std_ratio:
            logger.warning(
                f"High variance in scale estimates "
                f"(std/mean = {std_ratio:.2%} > {max_std_ratio:.2%})"
            )
            return False

        if scale_factor <= 0:
            logger.error("Invalid scale factor (must be positive)")
            return False

        if scale_factor < 0.01 or scale_factor > 100:
            logger.warning(
                f"Unusual scale factor {scale_factor:.4f} "
                f"(expected range: 0.01-100)"
            )

        return True