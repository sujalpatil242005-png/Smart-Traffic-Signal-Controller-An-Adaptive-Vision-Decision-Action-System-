"""
Configuration module for Smart Traffic Signal Controller
Defines vehicle weights, timing constraints, and system parameters
"""

# Vehicle Weight Mapping (based on COCO classes used by YOLO)
# Weights represent space consumption and inertia impact
VEHICLE_WEIGHTS = {
    'car': 1.0,           # Standard passenger car
    'motorcycle': 0.3,    # Takes less space, quick acceleration
    'bus': 3.0,           # Large vehicle, slow acceleration
    'truck': 2.5,         # Heavy vehicle
    'bicycle': 0.2,       # Minimal space
}

# COCO class IDs for vehicles
VEHICLE_CLASS_IDS = {
    2: 'car',
    3: 'motorcycle', 
    5: 'bus',
    7: 'truck',
    1: 'bicycle',
}

# Traffic Signal Timing Constraints
TOTAL_CYCLE_TIME = 120  # Total time for one complete cycle (seconds)
MIN_GREEN_TIME = 10     # Minimum green light duration (seconds)
MAX_GREEN_TIME = 60     # Maximum green light duration (seconds)

# YOLO Model Configuration
YOLO_MODEL = "yolov8n.pt"  # Nano model for fast inference
YOLO_INPUT_SIZE = 640       # Standard YOLO input resolution
CONFIDENCE_THRESHOLD = 0.25 # Minimum confidence for detections

# Road Names
ROAD_NAMES = ['North', 'South', 'East', 'West']

# Visualization Settings
HUD_BACKGROUND_COLOR = (0, 0, 0)        # Black
HUD_TEXT_COLOR = (255, 255, 255)        # White
HUD_HEIGHT = 50                          # Pixels
BBOX_HIGH_WEIGHT_COLOR = (0, 0, 255)    # Red for buses/trucks
BBOX_LOW_WEIGHT_COLOR = (0, 255, 0)     # Green for cars/bikes
BBOX_THICKNESS = 2
FONT = 0  # cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
FONT_THICKNESS = 2
