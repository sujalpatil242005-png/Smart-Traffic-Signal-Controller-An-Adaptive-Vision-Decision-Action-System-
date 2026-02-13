# Smart Traffic Signal Controller

An adaptive traffic signal system that uses YOLOv8 (or demo data) to estimate vehicle density per road, then allocates green time dynamically using a weighted scoring model.

## Project Overview

The system follows a closed-loop Vision-Decision-Action workflow:

1. **Vision**: Detect vehicles per road from images (or demo JSON data).
2. **Decision**: Compute weighted density scores and allocate green time under safety constraints.
3. **Action**: Render a crossroads dashboard and print analytics in the terminal.

## System Flow (Step-by-Step)

1. Read the four road images (`north.jpg`, `south.jpg`, `east.jpg`, `west.jpg`).
2. Detect vehicles on each road (YOLO) or load vehicles from `demo_data.json` (demo mode).
3. Convert detections into weighted density scores.
4. Compute green times from scores and clamp them to min/max constraints.
5. Assemble the crossroads dashboard and save `traffic_dashboard.jpg`.
6. Print all analytics to the terminal.

## Core Logic

### Weighted Density Score

Each road gets a density score based on vehicle type weights:

$$S_j = \sum_{i=1}^n (V_i \times W_i)$$

Where:
- $V_i$ is the detected vehicle i (confidence included)
- $W_i$ is the vehicle class weight

Default weights (see [config.py](config.py)):
- Car: 1.0
- Motorcycle: 0.3
- Bus: 3.0
- Truck: 2.5
- Bicycle: 0.2

### Green Time Allocation

$$G_j = \frac{S_j}{\sum_{k=1}^4 S_k} \times T_{total}$$

Constraints:
- Minimum green: 10s
- Maximum green: 60s
- Total cycle: 120s

If a road exceeds max, excess time is redistributed proportionally to the others.

## Dashboard Orientation Rules

The crossroads layout uses a 3x3 grid. Each road image is oriented as:
- **North**: rotated clockwise 90 degrees
- **East**: no change
- **West**: vertically mirrored (left-right flip)
- **South**: rotated counterclockwise 90 degrees

The dashboard is clean (no overlays) and saved as `traffic_dashboard.jpg`.

## Key Functions (What They Do)

### [main.py](main.py)

- `check_images_exist(image_dir)`: Verifies required images exist.
- `get_image_paths(image_dir)`: Resolves image paths with flexible extensions.
- `main()`: Parses `--demo`, loads images, runs the controller, renders output.

### [traffic_controller.py](traffic_controller.py)

- `__init__(model_path, demo_mode)`: Loads YOLO or demo JSON data.
- `detect_vehicles(image_path, road_name)`: Returns vehicle detections for one image.
- `calculate_density_score(vehicles)`: Computes weighted score.
- `allocate_green_times(density_scores)`: Clamps and redistributes green time.
- `draw_detections(image, vehicles)`: (Not used in output) Draws bounding boxes.
- `draw_hud(image, ...)`: (Not used in output) Adds HUD overlay.
- `create_dashboard(images)`: Builds crossroads layout with correct rotations.
- `process_intersection(image_paths)`: Orchestrates full loop and returns analytics.

### [config.py](config.py)

- Vehicle class weights
- COCO class IDs
- Timing constraints and model settings
- Visual styling constants

## Project Structure

```
dnn-sujal-project/
├── main.py                     # Entry point
├── traffic_controller.py       # Controller logic
├── config.py                   # Config and constants
├── demo_data.json              # Demo vehicle counts
├── create_demo_from_images.py  # Generate demo_data from YOLO
├── generate_images.py          # Synthetic image generator
├── get_real_images.py          # Guidance to get real images
├── images/                     # Input images
│   ├── north.jpg
│   ├── south.jpg
│   ├── east.jpg
│   └── west.jpg
├── pyproject.toml
└── README.md
```

## Installation

This project uses UV to manage Python dependencies:

```bash
uv sync
```

Dependencies:
- `ultralytics`
- `opencv-python`
- `numpy`

## Usage

### A) Normal Mode (YOLO detection)

1. Add real traffic photos to [images/](images/): `north.jpg`, `south.jpg`, `east.jpg`, `west.jpg`.
2. Run:

```bash
uv run main.py
```

### B) Demo Mode (skip detection)

Demo mode uses `demo_data.json` to simulate vehicle detection:

```bash
uv run main.py --demo
```

Environment variable alternative:

```bash
$env:DEMO_MODE="1"; uv run main.py
```

### C) Generate Demo Data From Real Images

```bash
uv run create_demo_from_images.py
```

This updates `demo_data.json` with actual YOLO detections.

## Example Terminal Output

```
============================================================
TRAFFIC SIGNAL CONTROLLER - PROCESSING CYCLE
============================================================

[SENSE] Processing North Road: images/north.jpg
  ├─ Detected: 12 vehicles
  └─ Density Score: 14.56

[DECIDE] Allocating green times...

Road       Score      Vehicles     Green Time
--------------------------------------------------
North      14.56      12           39.0s
South      3.03       5            9.9s
East       9.63       8            25.8s
West       16.94      15           45.4s
```

## Configuration Reference

Key settings in [config.py](config.py):

```python
TOTAL_CYCLE_TIME = 120
MIN_GREEN_TIME = 10
MAX_GREEN_TIME = 60
CONFIDENCE_THRESHOLD = 0.25
YOLO_MODEL = "yolov8n.pt"
```

## Tips and Troubleshooting

- If YOLO detects 0 vehicles, your images are likely synthetic. Use real photos or demo mode.
- If the dashboard looks incorrect, verify image orientation and sizes.
- If the model is slow, use `yolov8n.pt` (default) or lower the input size.

## License

MIT License