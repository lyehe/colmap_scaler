"""Triangulate marker corners from 2D detections to 3D points."""

import logging
import numpy as np
import pycolmap

logger = logging.getLogger(__name__)


class CornerTriangulator:
    """Triangulate marker corners from 2D detections."""

    def __init__(self, reconstruction: pycolmap.Reconstruction) -> None:
        """Initialize triangulator.

        :param reconstruction: COLMAP reconstruction.
        """
        self.reconstruction = reconstruction
        self.image_name_to_id = {
            img.name: img_id for img_id, img in reconstruction.images.items()
        }

    def triangulate_marker_corners(
        self, detections: dict, min_num_views: int = 2
    ) -> dict[int, dict]:
        """Triangulate all marker corners from 2D detections.

        :param detections: Detection results from JSON (detections dict).
        :param min_num_views: Minimum number of views required to triangulate.
        :returns: Dictionary mapping marker_id to corner data.
        """
        # Organize observations by marker ID and corner index
        marker_observations = {}

        for image_name, detection_data in detections.items():
            if image_name not in self.image_name_to_id:
                logger.warning(f"Image {image_name} not found in reconstruction")
                continue

            image_id = self.image_name_to_id[image_name]

            for marker in detection_data.get("markers", []):
                marker_id = marker["id"]
                corners = marker["corners"]

                if marker_id not in marker_observations:
                    marker_observations[marker_id] = {i: [] for i in range(4)}

                # Add observation for each corner
                for corner_idx, corner_2d in enumerate(corners):
                    marker_observations[marker_id][corner_idx].append(
                        {
                            "image_id": image_id,
                            "point2D": np.array(corner_2d, dtype=np.float64),
                        }
                    )

        # Triangulate each marker's corners
        results = {}
        for marker_id, corners_obs in marker_observations.items():
            corner_3d_points = []
            num_observations = []
            reprojection_errors = []
            valid = True

            for corner_idx in range(4):
                observations = corners_obs[corner_idx]

                if len(observations) < min_num_views:
                    logger.warning(
                        f"Marker {marker_id} corner {corner_idx} "
                        f"has only {len(observations)} views (need {min_num_views})"
                    )
                    valid = False
                    break

                # Triangulate this corner
                point3D, error = self._triangulate_point(observations)

                if point3D is None:
                    logger.warning(
                        f"Failed to triangulate marker {marker_id} corner {corner_idx}"
                    )
                    valid = False
                    break

                corner_3d_points.append(point3D.tolist())
                num_observations.append(len(observations))
                reprojection_errors.append(error)

            if valid:
                results[marker_id] = {
                    "corners_3d": corner_3d_points,
                    "num_observations": num_observations,
                    "reprojection_errors": reprojection_errors,
                    "valid": True,
                }
            else:
                results[marker_id] = {"valid": False}

        return results

    def _triangulate_point(
        self, observations: list[dict]
    ) -> tuple[np.ndarray | None, float]:
        """Triangulate a single 3D point from multiple 2D observations.

        :param observations: List of observations with image_id and point2D.
        :returns: Tuple of (3D point as numpy array, reprojection error).
        """
        if len(observations) < 2:
            return None, float("inf")

        # Collect camera poses, cameras, and undistorted normalized points
        points2D_normalized = []
        proj_matrices_normalized = []
        cameras = []
        images = []

        for obs in observations:
            image_id = obs["image_id"]
            image = self.reconstruction.images[image_id]
            camera = self.reconstruction.cameras[image.camera_id]

            # Undistort 2D point to normalized camera coordinates
            # This removes lens distortion and converts to normalized camera frame
            point2D_distorted = obs["point2D"]
            point2D_normalized = camera.cam_from_img(point2D_distorted)

            # Get camera pose (world to camera)
            cam_from_world = image.cam_from_world()
            R = cam_from_world.rotation.matrix()
            t = cam_from_world.translation

            # Create projection matrix P = [R|t] (no K, using normalized coordinates)
            Rt = np.hstack([R, t.reshape(3, 1)])

            proj_matrices_normalized.append(Rt)
            points2D_normalized.append(point2D_normalized)
            cameras.append(camera)
            images.append(image)

        # Triangulate using DLT with normalized coordinates
        point3D = self._triangulate_dlt(proj_matrices_normalized, points2D_normalized)

        if point3D is None:
            return None, float("inf")

        # Calculate reprojection error (using distortion model)
        error = self._calculate_reprojection_error_with_distortion(
            point3D, images, cameras, observations
        )

        return point3D, error

    def _triangulate_dlt(
        self, proj_matrices: list[np.ndarray], points2D: list[np.ndarray]
    ) -> np.ndarray | None:
        """Triangulate using Direct Linear Transform.

        :param proj_matrices: List of 3x4 projection matrices.
        :param points2D: List of 2D points.
        :returns: 3D point as numpy array or None if failed.
        """
        # Build matrix A for homogeneous linear system
        A = []
        for P, point2D in zip(proj_matrices, points2D):
            x, y = point2D
            A.append(x * P[2] - P[0])
            A.append(y * P[2] - P[1])

        A = np.array(A)

        # Solve using SVD
        try:
            _, _, Vt = np.linalg.svd(A)
            X = Vt[-1]
            X = X / X[3]  # Convert from homogeneous to 3D
            return X[:3]
        except np.linalg.LinAlgError:
            return None

    def _calculate_reprojection_error_with_distortion(
        self,
        point3D: np.ndarray,
        images: list,
        cameras: list,
        observations: list[dict],
    ) -> float:
        """Calculate mean reprojection error with proper distortion handling.

        :param point3D: 3D point in world coordinates.
        :param images: List of COLMAP images.
        :param cameras: List of COLMAP cameras.
        :param observations: List of observation dicts with original distorted points.
        :returns: Mean reprojection error in pixels.
        """
        errors = []

        for image, camera, obs in zip(images, cameras, observations):
            # Transform 3D point to camera coordinates
            cam_from_world = image.cam_from_world()
            point3D_cam = cam_from_world * point3D

            # Check if point is behind camera
            if point3D_cam[2] <= 0:
                # Point behind camera
                errors.append(float("inf"))
                continue

            # Apply distortion and convert to pixel coordinates (using 3D point)
            point2D_proj = camera.img_from_cam(point3D_cam)

            # Compare with original distorted observation
            point2D_obs = obs["point2D"]
            error = np.linalg.norm(point2D_proj - point2D_obs)
            errors.append(error)

        return np.mean(errors)

    def validate_triangulated_corners(
        self, triangulated_markers: dict[int, dict], max_reprojection_error: float = 3.0
    ) -> dict[int, dict]:
        """Validate triangulated corners and filter by reprojection error.

        :param triangulated_markers: Triangulation results.
        :param max_reprojection_error: Maximum allowed reprojection error in pixels.
        :returns: Filtered triangulation results.
        """
        validated = {}

        for marker_id, data in triangulated_markers.items():
            if not data.get("valid", False):
                continue

            # Check reprojection errors
            max_error = max(data["reprojection_errors"])
            mean_error = np.mean(data["reprojection_errors"])

            if max_error > max_reprojection_error:
                logger.warning(
                    f"Marker {marker_id} has high reprojection error: "
                    f"max={max_error:.2f}px, mean={mean_error:.2f}px"
                )
                continue

            validated[marker_id] = data

        return validated
