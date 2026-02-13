"""
Smart Traffic Signal Controller
Implements the Vision-Decision-Action loop for adaptive traffic management
"""

import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Tuple, Dict
import config
import json
from pathlib import Path


class TrafficController:
    """
    Main controller class that handles:
    1. Vehicle detection using YOLOv8
    2. Density score calculation with weighted vehicle counts
    3. Adaptive green time allocation with constraints
    4. Visualization with HUD overlays
    """
    
    def __init__(self, model_path: str = config.YOLO_MODEL, demo_mode: bool = False):
        """Initialize the traffic controller with YOLO model"""
        self.demo_mode = demo_mode
        self.demo_data = None
        
        if demo_mode:
            print(f"[INIT] Running in DEMO MODE - using demo_data.json")
            demo_file = Path(__file__).parent / "demo_data.json"
            if demo_file.exists():
                with open(demo_file, 'r') as f:
                    self.demo_data = json.load(f)
                print(f"[INIT] Loaded demo data with vehicle counts")
            else:
                print(f"[WARNING] demo_data.json not found, falling back to YOLO")
                self.demo_mode = False
        
        if not self.demo_mode:
            print(f"[INIT] Loading YOLO model: {model_path}")
            self.model = YOLO(model_path)
            
        self.road_names = config.ROAD_NAMES
        print("[INIT] Traffic Controller initialized successfully")
    
    def detect_vehicles(self, image_path: str, road_name: str = None) -> Tuple[np.ndarray, List[Dict]]:
        """
        Detect vehicles in an image and return annotated image + detections
        In demo mode, uses pre-defined counts from demo_data.json
        
        Args:
            image_path: Path to the input image
            road_name: Name of the road (for demo mode)
            
        Returns:
            Tuple of (annotated_image, list of vehicle detections)
        """
        # Load and preprocess image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        vehicles = []
        
        # Demo mode: use JSON data instead of YOLO
        if self.demo_mode and self.demo_data and road_name:
            road_key = road_name.lower()
            if road_key in self.demo_data.get('roads', {}):
                demo_vehicles = self.demo_data['roads'][road_key]['vehicles']
                h, w = image.shape[:2]
                
                # Generate fake bounding boxes for visualization
                for i, v_data in enumerate(demo_vehicles):
                    vehicle_type = v_data['type']
                    confidence = v_data['confidence']
                    weight = config.VEHICLE_WEIGHTS.get(vehicle_type, 1.0)
                    
                    # Create a fake bbox spread across the image
                    x_offset = (i % 4) * (w // 4) + 20
                    y_offset = (i // 4) * (h // 6) + 50
                    bbox_w = 80 if vehicle_type in ['bus', 'truck'] else 60
                    bbox_h = 60 if vehicle_type in ['bus', 'truck'] else 45
                    
                    bbox = np.array([x_offset, y_offset, x_offset + bbox_w, y_offset + bbox_h])
                    
                    vehicles.append({
                        'type': vehicle_type,
                        'weight': weight,
                        'confidence': confidence,
                        'bbox': bbox
                    })
        else:
            # Real YOLO inference
            results = self.model(image, conf=config.CONFIDENCE_THRESHOLD, verbose=False)
            
            # Extract vehicle detections
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    class_id = int(box.cls[0])
                    
                    # Filter only vehicles
                    if class_id in config.VEHICLE_CLASS_IDS:
                        vehicle_type = config.VEHICLE_CLASS_IDS[class_id]
                        weight = config.VEHICLE_WEIGHTS[vehicle_type]
                        confidence = float(box.conf[0])
                        bbox = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
                        
                        vehicles.append({
                            'type': vehicle_type,
                            'weight': weight,
                            'confidence': confidence,
                            'bbox': bbox
                        })
        
        return image, vehicles
    
    def calculate_density_score(self, vehicles: List[Dict]) -> float:
        """
        Calculate weighted density score for a road
        Formula: S_j = Σ(V_i × W_i)
        
        Args:
            vehicles: List of detected vehicles with weights
            
        Returns:
            Density score (float)
        """
        score = sum(v['weight'] * v['confidence'] for v in vehicles)
        return round(score, 2)
    
    def allocate_green_times(self, density_scores: List[float]) -> List[float]:
        """
        Allocate green light times based on density scores with constraints
        
        Algorithm:
        1. Calculate proportional times: G_j = (S_j / Σ S_k) × T_total
        2. Apply min/max constraints (10s - 60s)
        3. Redistribute excess time from capped roads
        
        Args:
            density_scores: List of 4 density scores [North, South, East, West]
            
        Returns:
            List of 4 green times in seconds
        """
        total_score = sum(density_scores)
        
        # Handle edge case: all roads empty
        if total_score == 0:
            return [config.TOTAL_CYCLE_TIME / 4] * 4
        
        # Step 1: Calculate raw proportional times
        raw_times = [
            (score / total_score) * config.TOTAL_CYCLE_TIME 
            for score in density_scores
        ]
        
        # Step 2: Apply constraints
        green_times = []
        excess_time = 0
        capped_indices = []
        
        for i, time in enumerate(raw_times):
            if time < config.MIN_GREEN_TIME:
                green_times.append(config.MIN_GREEN_TIME)
                excess_time -= (config.MIN_GREEN_TIME - time)
            elif time > config.MAX_GREEN_TIME:
                green_times.append(config.MAX_GREEN_TIME)
                excess_time += (time - config.MAX_GREEN_TIME)
                capped_indices.append(i)
            else:
                green_times.append(time)
        
        # Step 3: Redistribute excess time
        if excess_time > 0:
            # Distribute to uncapped roads proportionally
            uncapped_indices = [i for i in range(4) if i not in capped_indices]
            if uncapped_indices:
                uncapped_scores = sum(density_scores[i] for i in uncapped_indices)
                if uncapped_scores > 0:
                    for i in uncapped_indices:
                        proportion = density_scores[i] / uncapped_scores
                        green_times[i] += excess_time * proportion
                        # Ensure still within max constraint
                        green_times[i] = min(green_times[i], config.MAX_GREEN_TIME)
        
        # Normalize to ensure total equals TOTAL_CYCLE_TIME
        actual_total = sum(green_times)
        green_times = [
            round((t / actual_total) * config.TOTAL_CYCLE_TIME, 1) 
            for t in green_times
        ]
        
        return green_times
    
    def draw_hud(self, image: np.ndarray, road_name: str, 
                 score: float, green_time: float, 
                 vehicle_count: int) -> np.ndarray:
        """
        Draw Heads-Up Display overlay on image
        
        Args:
            image: Input image
            road_name: Name of the road (North/South/East/West)
            score: Density score
            green_time: Allocated green time in seconds
            vehicle_count: Number of detected vehicles
            
        Returns:
            Image with HUD overlay
        """
        h, w = image.shape[:2]
        
        # Create semi-transparent black bar
        overlay = image.copy()
        cv2.rectangle(overlay, (0, 0), (w, config.HUD_HEIGHT), 
                     config.HUD_BACKGROUND_COLOR, -1)
        image = cv2.addWeighted(overlay, 0.7, image, 0.3, 0)
        
        # Draw text
        status = "🟢 GREEN" if green_time >= config.MIN_GREEN_TIME else "🔴 LOW"
        text = f"{road_name} | SCORE: {score} | TIME: {green_time}s | VEHICLES: {vehicle_count}"
        
        cv2.putText(image, text, (10, 30), config.FONT, config.FONT_SCALE,
                   config.HUD_TEXT_COLOR, config.FONT_THICKNESS)
        
        return image
    
    def draw_detections(self, image: np.ndarray, vehicles: List[Dict]) -> np.ndarray:
        """
        Draw bounding boxes on detected vehicles
        Color coding: Red for high-weight vehicles (bus/truck), Green for low-weight
        
        Args:
            image: Input image
            vehicles: List of vehicle detections
            
        Returns:
            Image with bounding boxes
        """
        for vehicle in vehicles:
            bbox = vehicle['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            
            # Choose color based on weight
            color = (config.BBOX_HIGH_WEIGHT_COLOR if vehicle['weight'] >= 2.0 
                    else config.BBOX_LOW_WEIGHT_COLOR)
            
            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, config.BBOX_THICKNESS)
            
            # Draw label
            label = f"{vehicle['type']} {vehicle['confidence']:.2f}"
            label_size = cv2.getTextSize(label, config.FONT, 0.5, 1)[0]
            cv2.rectangle(image, (x1, y1 - label_size[1] - 5), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(image, label, (x1, y1 - 5), config.FONT, 0.5, 
                       (255, 255, 255), 1)
        
        return image
    
    def create_dashboard(self, images: List[np.ndarray]) -> np.ndarray:
        """
        Create intersection-style dashboard where roads meet at crossroads
        Layout (Crossroads View):
               [ North ]
        [West] [Center] [East]
               [ South ]
        
        Args:
            images: List of 4 images [North, South, East, West]
            
        Returns:
            Combined intersection-style dashboard
        """
        # Use square sections for simplicity - all 400x400
        section_size = 400
        
        # Resize and rotate images as specified
        north_img = cv2.resize(images[0], (section_size, section_size))
        north_img = cv2.rotate(north_img, cv2.ROTATE_90_CLOCKWISE)  # North: clockwise 90°
        
        south_img = cv2.resize(images[1], (section_size, section_size))
        south_img = cv2.rotate(south_img, cv2.ROTATE_90_COUNTERCLOCKWISE)  # South: anticlockwise 90°
        
        # East: no changes
        east_img = cv2.resize(images[3], (section_size, section_size))
        
        # West: vertically mirrored (flip left-right)
        west_img = cv2.resize(images[2], (section_size, section_size))
        west_img = cv2.flip(west_img, 1)  # flipCode=1 for vertical mirror (left-right flip)
        
        # Create center intersection visualization with road extensions
        center = np.zeros((section_size, section_size, 3), dtype=np.uint8)
        center[:] = (60, 60, 60)  # Dark gray for road surface
        
        # Draw wider roads extending to edges (to connect with incoming roads)
        road_width = 200  # Wide enough to visually connect
        road_color = (70, 70, 70)
        
        # Horizontal road (West-East)
        cv2.rectangle(center, (0, section_size//2 - road_width//2), 
                     (section_size, section_size//2 + road_width//2), road_color, -1)
        
        # Vertical road (North-South)
        cv2.rectangle(center, (section_size//2 - road_width//2, 0), 
                     (section_size//2 + road_width//2, section_size), road_color, -1)
        
        # Draw center lines (yellow dashed)
        dash_color = (0, 200, 255)
        dash_length = 30
        dash_gap = 20
        
        # Horizontal center line
        for x in range(0, section_size, dash_length + dash_gap):
            cv2.line(center, (x, section_size//2), 
                    (min(x + dash_length, section_size), section_size//2), dash_color, 3)
        
        # Vertical center line
        for y in range(0, section_size, dash_length + dash_gap):
            cv2.line(center, (section_size//2, y), 
                    (section_size//2, min(y + dash_length, section_size)), dash_color, 3)
        
        # Draw crosswalk stripes (zebra crossings)
        stripe_color = (255, 255, 255)
        stripe_width = 15
        stripe_gap = 20
        offset = 120  # Distance from center
        crosswalk_width = 80
        
        # Horizontal crosswalks (top and bottom)
        for i in range(section_size//2 - crosswalk_width//2, section_size//2 + crosswalk_width//2, stripe_gap):
            # Top crosswalk
            cv2.rectangle(center, (i, section_size//2 - offset - 40), 
                         (i + stripe_width, section_size//2 - offset), stripe_color, -1)
            # Bottom crosswalk
            cv2.rectangle(center, (i, section_size//2 + offset), 
                         (i + stripe_width, section_size//2 + offset + 40), stripe_color, -1)
        
        # Vertical crosswalks (left and right)
        for i in range(section_size//2 - crosswalk_width//2, section_size//2 + crosswalk_width//2, stripe_gap):
            # Left crosswalk
            cv2.rectangle(center, (section_size//2 - offset - 40, i), 
                         (section_size//2 - offset, i + stripe_width), stripe_color, -1)
            # Right crosswalk
            cv2.rectangle(center, (section_size//2 + offset, i), 
                         (section_size//2 + offset + 40, i + stripe_width), stripe_color, -1)
        
        # Create 3x3 canvas
        pad_size = 40
        canvas_size = section_size * 3 + pad_size * 4
        canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
        canvas[:] = (30, 30, 30)  # Dark background
        
        # Calculate positions in 3x3 grid
        positions = {}
        for row in range(3):
            for col in range(3):
                x = pad_size + col * (section_size + pad_size)
                y = pad_size + row * (section_size + pad_size)
                positions[f"_{row}_{col}"] = (y, x)
        
        # Place images in crossroads formation
        # North (top center) - row 0, col 1
        y, x = positions["_0_1"]
        canvas[y:y + section_size, x:x + section_size] = north_img
        
        # West (middle left) - row 1, col 0
        y, x = positions["_1_0"]
        canvas[y:y + section_size, x:x + section_size] = west_img
        
        # Center (middle center) - row 1, col 1
        y, x = positions["_1_1"]
        canvas[y:y + section_size, x:x + section_size] = center
        
        # East (middle right) - row 1, col 2
        y, x = positions["_1_2"]
        canvas[y:y + section_size, x:x + section_size] = east_img
        
        # South (bottom center) - row 2, col 1
        y, x = positions["_2_1"]
        canvas[y:y + section_size, x:x + section_size] = south_img
        
        # Add directional labels
        label_color = (255, 255, 255)
        font_scale = 1.2
        thickness = 3
        
        # North label (above north section)
        y_n, x_n = positions["_0_1"]
        cv2.putText(canvas, "NORTH", (x_n + section_size//3, y_n - 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, label_color, thickness)
        
        # South label (below south section)
        y_s, x_s = positions["_2_1"]
        cv2.putText(canvas, "SOUTH", (x_s + section_size//3, y_s + section_size + 45), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, label_color, thickness)
        
        # West label (left of west section)
        y_w, x_w = positions["_1_0"]
        cv2.putText(canvas, "WEST", (x_w - 10, y_w - 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, label_color, thickness)
        
        # East label (right of east section)
        y_e, x_e = positions["_1_2"]
        cv2.putText(canvas, "EAST", (x_e + section_size - 110, y_e - 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, label_color, thickness)
        
        return canvas
    
    def process_intersection(self, image_paths: List[str]) -> Tuple[np.ndarray, Dict]:
        """
        Complete Vision-Decision-Action cycle for the intersection
        
        Args:
            image_paths: List of 4 image paths [North, South, East, West]
            
        Returns:
            Tuple of (dashboard_image, analytics_dict)
        """
        print("\n" + "="*60)
        print("TRAFFIC SIGNAL CONTROLLER - PROCESSING CYCLE")
        print("="*60)
        
        # Phase 1: SENSE - Detect vehicles on all roads
        all_vehicles = []
        all_images = []
        density_scores = []
        
        for i, (road_name, img_path) in enumerate(zip(self.road_names, image_paths)):
            print(f"\n[SENSE] Processing {road_name} Road: {img_path}")
            image, vehicles = self.detect_vehicles(img_path, road_name)
            score = self.calculate_density_score(vehicles)
            
            all_vehicles.append(vehicles)
            all_images.append(image)
            density_scores.append(score)
            
            print(f"  ├─ Detected: {len(vehicles)} vehicles")
            print(f"  └─ Density Score: {score}")
        
        # Phase 2: DECIDE - Allocate green times
        print(f"\n[DECIDE] Allocating green times...")
        green_times = self.allocate_green_times(density_scores)
        
        print(f"\n{'Road':<10} {'Score':<10} {'Vehicles':<12} {'Green Time':<12}")
        print("-" * 50)
        for road, score, vehicles, time in zip(
            self.road_names, density_scores, all_vehicles, green_times
        ):
            print(f"{road:<10} {score:<10} {len(vehicles):<12} {time:<12}s")
        
# Phase 3: VISUALIZE - Create dashboard with Bounding Boxes and HUD labels
        print(f"\n[VISUALIZE] Creating dashboard with vehicle classifications...")
        annotated_images = []
        
        for i, (image, vehicles, road_name, score, green_time) in enumerate(
            zip(all_images, all_vehicles, self.road_names, density_scores, green_times)
        ):
            # 1. Start with a copy of the image
            ann_img = image.copy()
            
            # 2. NEW: Draw Bounding Boxes and Vehicle Names
            # This calls the existing draw_detections method in this class
            ann_img = self.draw_detections(ann_img, vehicles)
            
            # 3. Determine Density Label and Color for HUD
            count = len(vehicles)
            if count <= 4:
                density_label, color = "LIGHT", (0, 255, 0)      # Green
            elif count <= 10:
                density_label, color = "MODERATE", (0, 255, 255)    # Yellow
            else:
                density_label, color = "HEAVY", (0, 0, 255)      # Red

            # 4. Add HUD Background and Text
            cv2.rectangle(ann_img, (0, 0), (ann_img.shape[1], 60), (0, 0, 0), -1)
            
            title_text = f"{road_name.upper()}: {density_label}"
            count_text = f"Count: {count} Veh"
            
            cv2.putText(ann_img, title_text, (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            cv2.putText(ann_img, count_text, (ann_img.shape[1] - 250, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            
            annotated_images.append(ann_img)
        
        # Create final dashboard
        dashboard = self.create_dashboard(annotated_images)
        
        # Prepare analytics
        analytics = {
            'density_scores': density_scores,
            'green_times': green_times,
            'vehicle_counts': [len(v) for v in all_vehicles],
            'total_vehicles': sum(len(v) for v in all_vehicles)
        }
        
        print(f"\n[COMPLETE] Total vehicles detected: {analytics['total_vehicles']}")
        print("="*60 + "\n")
        
        return dashboard, analytics
