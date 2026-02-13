"""
Generate Sample Traffic Images for Testing
Creates synthetic traffic scenes with vehicles for the 4 roads
"""

import cv2
import numpy as np
from pathlib import Path


def create_traffic_scene(road_name: str, num_vehicles: int, scene_type: str = "moderate") -> np.ndarray:
    """
    Create a synthetic traffic scene
    
    Args:
        road_name: Name of the road (North/South/East/West)
        num_vehicles: Number of vehicles to simulate
        scene_type: Traffic density (light/moderate/heavy)
    
    Returns:
        Image array
    """
    # Create base image (road)
    width, height = 1280, 720
    image = np.ones((height, width, 3), dtype=np.uint8) * 150  # Gray road
    
    # Draw road markings
    cv2.rectangle(image, (0, 0), (width, 100), (100, 200, 100), -1)  # Grass
    cv2.rectangle(image, (0, height-100), (width, height), (100, 200, 100), -1)  # Grass
    
    # Draw lane dividers
    for i in range(0, width, 100):
        cv2.rectangle(image, (i, height//2 - 2), (i + 50, height//2 + 2), (255, 255, 255), -1)
    
    # Add road name label
    cv2.putText(image, f"{road_name} ROAD", (50, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    
    # Simulate vehicles with colored rectangles
    colors = {
        'car': (50, 100, 200),      # Blue
        'bus': (200, 50, 50),       # Red
        'truck': (100, 100, 200),   # Purple
        'motorcycle': (50, 200, 50)  # Green
    }
    
    vehicles = []
    if scene_type == "light":
        vehicles = ['car'] * min(num_vehicles, 3)
    elif scene_type == "moderate":
        vehicles = ['car'] * (num_vehicles // 2) + ['bus'] * (num_vehicles // 4) + ['motorcycle'] * (num_vehicles // 4)
    else:  # heavy
        vehicles = ['car'] * (num_vehicles // 2) + ['bus'] * (num_vehicles // 3) + ['truck'] * (num_vehicles // 6)
    
    # Draw vehicles
    x_start = 100
    for i, vehicle_type in enumerate(vehicles[:15]):  # Max 15 visible vehicles
        x = x_start + i * 80
        y = 200 + (i % 3) * 120  # 3 lanes
        
        if vehicle_type == 'bus':
            w, h = 120, 70
        elif vehicle_type == 'truck':
            w, h = 100, 65
        elif vehicle_type == 'motorcycle':
            w, h = 40, 50
        else:  # car
            w, h = 70, 55
        
        color = colors.get(vehicle_type, (100, 100, 100))
        cv2.rectangle(image, (x, y), (x + w, y + h), color, -1)
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 255), 2)
        
        # Add wheels
        cv2.circle(image, (x + 15, y + h), 8, (0, 0, 0), -1)
        cv2.circle(image, (x + w - 15, y + h), 8, (0, 0, 0), -1)
    
    # Add traffic density indicator
    density_text = f"Vehicles: {num_vehicles} | Density: {scene_type.upper()}"
    cv2.putText(image, density_text, (50, height - 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return image


def main():
    """Generate sample images for all 4 roads"""
    print("\n" + "="*60)
    print("GENERATING SAMPLE TRAFFIC IMAGES")
    print("="*60 + "\n")
    
    # Create images directory
    project_dir = Path(__file__).parent
    image_dir = project_dir / "images"
    image_dir.mkdir(exist_ok=True)
    
    # Define scenarios for each road
    scenarios = {
        'north': {'num_vehicles': 12, 'scene_type': 'heavy'},
        'south': {'num_vehicles': 5, 'scene_type': 'light'},
        'east': {'num_vehicles': 8, 'scene_type': 'moderate'},
        'west': {'num_vehicles': 15, 'scene_type': 'heavy'},
    }
    
    # Generate images
    for road_name, params in scenarios.items():
        print(f"Generating {road_name}.jpg - {params['scene_type']} traffic ({params['num_vehicles']} vehicles)")
        
        image = create_traffic_scene(
            road_name.upper(), 
            params['num_vehicles'], 
            params['scene_type']
        )
        
        output_path = image_dir / f"{road_name}.jpg"
        cv2.imwrite(str(output_path), image)
        print(f"  ✓ Saved to: {output_path}")
    
    print("\n" + "="*60)
    print("✅ All sample images generated successfully!")
    print(f"📁 Location: {image_dir}")
    print("\nYou can now run: python main.py")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()