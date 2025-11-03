# COLMAP Scaler

Automatically rescale COLMAP reconstructions to real-world units using fiducial markers (ArUco/AprilTag).

## What It Does

COLMAP Scaler detects fiducial markers in your images, triangulates them in 3D space, and uses their known physical size to rescale your entire COLMAP reconstruction. The tool:

1. Detects ArUco or AprilTag markers in images
2. Undistorts detected corners using COLMAP camera distortion models
3. Triangulates marker corners to 3D points using camera parameters and poses
4. Estimates scale factor from known marker size vs. reconstructed size
5. Applies uniform scaling to the entire reconstruction (in-place with backup)

## Installation

```bash
uv pip install -e .
```

## Quick Start

```bash
# Minimal usage (auto-detects marker dictionary)
uv run colmap-scaler --input my_dataset --marker-size 0.05

# With specific dictionary
uv run colmap-scaler --input my_dataset --marker-size 0.05 --dictionary DICT_APRILTAG_25h9
```

**Requirements:**
- Dataset must have `images/` and `sparse/0/` folders
- Marker size must be in meters (e.g., 0.05 for 50mm markers)

## Usage Examples

### Basic Scaling

```bash
# AprilTag markers (50mm = 0.05m)
uv run colmap-scaler --input data/my_dataset --marker-size 0.05 --dictionary DICT_APRILTAG_25h9

# ArUco markers (150mm = 0.15m)
uv run colmap-scaler --input data/my_dataset --marker-size 0.15 --dictionary DICT_4X4_50
```

### With Visualization

```bash
# Save marker visualizations to dataset_folder/tags/
uv run colmap-scaler --input data/my_dataset --marker-size 0.05 --dictionary DICT_APRILTAG_25h9 --visualize
```

### Export JSON Reports

```bash
# Generate detections.json and scale_report.json
uv run colmap-scaler --input data/my_dataset --marker-size 0.05 --export-json
```

### Process Subset of Images

```bash
# Process only 20 random images
uv run colmap-scaler --input data/my_dataset --marker-size 0.05 --num-images 20
```

### Custom Reconstruction Path

```bash
# If reconstruction is in different location
uv run colmap-scaler --input data/images_only --marker-size 0.05 --reconstruction data/sparse_recon
```

### Custom Output Location

```bash
# Save scaled reconstruction to sparse/1 (preserves sparse/0)
uv run colmap-scaler --input data/my_dataset --marker-size 0.05 --output-name 1

# Save to custom folder name
uv run colmap-scaler --input data/my_dataset --marker-size 0.05 --output-name scaled --backup-name original
```

## Comprehensive Example

```bash
# Full pipeline with all options
uv run colmap-scaler \
    --input data/bed_scale \
    --marker-size 0.05 \
    --dictionary DICT_APRILTAG_25h9 \
    --visualize \
    --export-json \
    --min-num-views 3 \
    --max-reprojection-error 1.5
```

## CLI Options

### Required
- `--input PATH` - Dataset folder (must contain `images/` and `sparse/0/`)
- `--marker-size FLOAT` - Physical marker size in meters (e.g., 0.05 for 50mm)

### Detection Options
- `--dictionary STR` - Marker dictionary (e.g., DICT_APRILTAG_25h9). Auto-detect if not specified
- `--num-images INT` - Process only N random images
- `--image-list PATH` - Text file with specific images to process (one per line)
- `--visualize` - Save annotated images to `dataset_folder/tags/`
- `--refine-corners` - Apply subpixel refinement (default: True)
- `--min-perimeter INT` - Minimum marker perimeter in pixels (default: 50)
- `--max-perimeter INT` - Maximum marker perimeter in pixels (default: 10000)

### Scaling Options
- `--reconstruction PATH` - Path to COLMAP sparse reconstruction (default: dataset/sparse/0)
- `--min-num-views INT` - Minimum views to triangulate corner (default: 2)
- `--max-reprojection-error FLOAT` - Maximum reprojection error in pixels (default: 3.0)
- `--output-name STR` - Output folder name under sparse/ (default: 0)
- `--backup-name STR` - Backup folder name (default: {output_name}_before_scale)

### Output Options
- `--export-json` - Export detections.json and scale_report.json (default: False)

## Supported Marker Dictionaries

**ArUco:**
- DICT_4X4_50, DICT_4X4_100, DICT_4X4_250, DICT_4X4_1000
- DICT_5X5_50, DICT_5X5_100, DICT_5X5_250, DICT_5X5_1000
- DICT_6X6_50, DICT_6X6_100, DICT_6X6_250, DICT_6X6_1000
- DICT_7X7_50, DICT_7X7_100, DICT_7X7_250, DICT_7X7_1000
- DICT_ARUCO_ORIGINAL

**AprilTag:**
- DICT_APRILTAG_16h5
- DICT_APRILTAG_25h9
- DICT_APRILTAG_36h10
- DICT_APRILTAG_36h11

## Output Files

**Default (in-place scaling):**
```
dataset_folder/
├── images/              # Original images
├── tags/                # Visualization images (if --visualize)
├── sparse/
│   ├── 0/              # Scaled reconstruction (in-place)
│   └── 0_before_scale/ # Backup of original
├── scale.log           # Complete log report
├── detections.json     # Marker detections (if --export-json)
└── scale_report.json   # Scaling statistics (if --export-json)
```

**With custom output (--output-name 1):**
```
dataset_folder/
├── sparse/
│   ├── 0/              # Original (preserved)
│   └── 1/              # Scaled reconstruction (new)
└── scale.log
```

### scale.log Format

```
============================================================
COLMAP Scaler - Scale Log Report
============================================================
Timestamp: 2025-11-03T15:23:42.838256
Dataset: ../data/table_scale_sfm/table_scale

------------------------------------------------------------
PHASE 1: MARKER DETECTION
------------------------------------------------------------
Marker type: apriltag
Marker size: 50.0mm (0.050m)
Dictionary: DICT_APRILTAG_25h9
Total images processed: 18
Images with markers: 18
Detection rate: 100.0%
Unique markers: [0, 2, 3]
Average markers per image: 1.28

------------------------------------------------------------
PHASE 2: RECONSTRUCTION SCALING
------------------------------------------------------------
Original Reconstruction:
  Cameras: 20
  Images: 20
  3D Points: 6931
  Scene extent: 102.1058 units
  Bounding box: [32.7515, 34.8021, 90.2317]
Triangulated markers: 3/3
Scale estimation method: robust_mean
Markers used: [0, 2, 3]
Scale factor: 0.144516
...
Scaled Reconstruction:
  Scene extent: 14.7559 meters
  Bounding box: [4.7331, 5.0294, 13.0399] meters

Scale reduction factor: 6.9x
Before: 102.11 units -> After: 14.76 meters
============================================================
```

## How It Works

1. **Detection**: Detects fiducial markers in all images and extracts 2D corner coordinates
2. **Undistortion**: Removes lens distortion from detected corners using COLMAP camera models
3. **Triangulation**: Triangulates undistorted marker corners to 3D points using camera poses
4. **Scale Estimation**: Compares known marker size with reconstructed 3D distances
5. **Scaling**: Applies uniform similarity transformation to entire reconstruction
6. **Output**: Saves scaled reconstruction (creates backup if overwriting existing folder)

**Important Notes:**
- **Properly handles camera distortion**: Undistorts 2D detections before triangulation for accurate 3D reconstruction
- Camera intrinsics (focal length, principal point, distortion) remain unchanged
- Only 3D points and camera positions (extrinsics) are scaled
- Scale estimation uses robust mean method with outlier rejection
- Multiple markers improve accuracy and robustness
- Works with all COLMAP camera models (SIMPLE_PINHOLE, PINHOLE, SIMPLE_RADIAL, RADIAL, OPENCV, etc.)
- Use `--output-name` to write to new location and preserve original

## Requirements

- Python >= 3.12
- opencv-contrib-python >= 4.12.0
- pycolmap >= 3.12.6
- numpy >= 1.24.0
- tyro >= 0.8.0

## Project Structure

```
colmap_scaler/
├── src/colmap_scaler/
│   ├── cli.py              # CLI interface
│   ├── pipeline.py         # Main pipeline orchestration
│   ├── config.py           # Configuration dataclasses
│   ├── detector.py         # Marker detection
│   ├── triangulator.py     # 3D triangulation
│   ├── scale_estimator.py  # Scale calculation
│   ├── scaler.py           # Reconstruction scaling
│   └── utils.py            # Helper functions
├── pyproject.toml
└── README.md
```

## License
[Apache License 2.0](LICENSE)
