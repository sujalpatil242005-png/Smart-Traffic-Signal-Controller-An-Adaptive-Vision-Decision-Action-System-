"""
Download or create realistic traffic images for testing
Uses real traffic scene samples that YOLO can detect
"""

import cv2
import numpy as np
from pathlib import Path
import urllib.request


def download_sample_images():
    """Download sample traffic images from public sources"""
    project_dir = Path(__file__).parent
    image_dir = project_dir / "images"
    image_dir.mkdir(exist_ok=True)
    
    print("\n" + "="*60)
    print("DOWNLOADING SAMPLE TRAFFIC IMAGES")
    print("="*60 + "\n")
    
    # Sample traffic images (public domain / creative commons)
    # These are placeholder URLs - you would need actual traffic images
    samples = {
        'north': 'https://images.unsplash.com/photo-1449824913935-59a10b8d2000?w=800',  # City traffic
        'south': 'https://images.unsplash.com/photo-1486299267070-83823f5448dd?w=800',  # Highway
        'east': 'https://images.unsplash.com/photo-1502489597346-dad15683d4c2?w=800',   # City street
        'west': 'https://images.unsplash.com/photo-1459257868276-5e51b3e5787a?w=800',   # Urban traffic
    }
    
    print("NOTE: Using placeholder traffic scenes.")
    print("For best results, place your own traffic images in images/ folder")
    print("(north.jpg, south.jpg, east.jpg, west.jpg)\n")
    
    # For now, create better synthetic images with more realistic vehicle shapes
    create_realistic_synthetic_images()


def create_realistic_synthetic_images():
    """Create synthetic images with better vehicle representations"""
    project_dir = Path(__file__).parent
    image_dir = project_dir / "images"
    
    print("Creating improved synthetic traffic images...\n")
    
    scenarios = {
        'north': {'vehicles': 12, 'density': 'heavy'},
        'south': {'vehicles': 5, 'density': 'light'},
        'east': {'vehicles': 8, 'density': 'moderate'},
        'west': {'vehicles': 15, 'density': 'heavy'},
    }
    
    for road, params in scenarios.items():
        print(f"Creating {road}.jpg - {params['density']} traffic")
        
        # Create more realistic traffic scene
        image = create_realistic_scene(
            road.upper(), 
            params['vehicles'], 
            params['density']
        )
        
        output_path = image_dir / f"{road}.jpg"
        cv2.imwrite(str(output_path), image)
        print(f"  ✓ Saved to: {output_path}")
    
    print("\n" + "="*60)
    print("⚠️  IMPORTANT: These are synthetic images!")
    print("YOLO may not detect the simple shapes as vehicles.")
    print("\nFor REAL detection, please use actual traffic photos:")
    print("  1. Take photos of real traffic")
    print("  2. Download from traffic camera feeds")
    print("  3. Use Google Street View screenshots")
    print("  4. Get images from traffic datasets")
    print("="*60 + "\n")


def create_realistic_scene(road_name, num_vehicles, density):
    """Create a more realistic traffic scene with vehicle-like shapes"""
    width, height = 1280, 720
    
    # Create road background
    image = np.ones((height, width, 3), dtype=np.uint8) * 40  # Dark asphalt
    
    # Draw road with realistic texture
    for i in range(0, width, 200):
        noise = np.random.randint(-10, 10)
        gray = 60 + noise
        cv2.rectangle(image, (i, 150), (i + 100, height - 150), (gray, gray, gray), -1)
    
    # Draw lane markings (white dashed lines)
    for y in [height // 3, 2 * height // 3]:
        for x in range(0, width, 80):
            cv2.rectangle(image, (x, y - 2), (x + 40, y + 2), (255, 255, 255), -1)
    
    # Draw center line (yellow)
    for x in range(0, width, 80):
        cv2.rectangle(image, (x, height // 2 - 2), (x + 40, height // 2 + 2), (0, 200, 255), -1)
    
    # Road edges
    cv2.rectangle(image, (0, 100), (width, 150), (80, 120, 80), -1)  # Grass/curb
    cv2.rectangle(image, (0, height - 150), (width, height - 100), (80, 120, 80), -1)
    
    # Add road sign
    cv2.putText(image, f"{road_name} ROAD - {density.upper()} TRAFFIC", 
                (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    
    # Draw more realistic vehicle shapes
    np.random.seed(42)  # For consistency
    
    for i in range(num_vehicles):
        lane = i % 3  # 3 lanes
        x = 100 + (i * 70) % (width - 200)
        y = 200 + lane * 150
        
        # Randomize vehicle type
        vehicle_type = np.random.choice(['car', 'car', 'car', 'bus', 'truck'], p=[0.5, 0.2, 0.1, 0.1, 0.1])
        
        if vehicle_type == 'bus':
            draw_bus(image, x, y)
        elif vehicle_type == 'truck':
            draw_truck(image, x, y)
        else:
            draw_car(image, x, y)
    
    # Add info overlay
    info = f"Vehicles: {num_vehicles} | Density: {density.upper()}"
    cv2.rectangle(image, (0, height - 80), (width, height - 100), (0, 0, 0), -1)
    cv2.putText(image, info, (50, height - 110), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return image


def draw_car(image, x, y):
    """Draw a car-like shape"""
    # Main body
    cv2.rectangle(image, (x, y), (x + 80, y + 50), (30, 60, 150), -1)
    cv2.rectangle(image, (x, y), (x + 80, y + 50), (255, 255, 255), 2)
    
    # Windshield
    pts = np.array([[x + 20, y + 10], [x + 60, y + 10], [x + 70, y + 25], [x + 10, y + 25]], np.int32)
    cv2.fillPoly(image, [pts], (100, 150, 200))
    
    # Wheels
    cv2.circle(image, (x + 20, y + 50), 10, (20, 20, 20), -1)
    cv2.circle(image, (x + 60, y + 50), 10, (20, 20, 20), -1)
    cv2.circle(image, (x + 20, y + 50), 6, (50, 50, 50), -1)
    cv2.circle(image, (x + 60, y + 50), 6, (50, 50, 50), -1)


def draw_bus(image, x, y):
    """Draw a bus-like shape"""
    # Main body (larger)
    cv2.rectangle(image, (x, y), (x + 140, y + 70), (200, 50, 50), -1)
    cv2.rectangle(image, (x, y), (x + 140, y + 70), (255, 255, 255), 2)
    
    # Windows
    for wx in range(x + 15, x + 130, 30):
        cv2.rectangle(image, (wx, y + 15), (wx + 20, y + 35), (150, 200, 255), -1)
    
    # Wheels
    cv2.circle(image, (x + 30, y + 70), 12, (20, 20, 20), -1)
    cv2.circle(image, (x + 110, y + 70), 12, (20, 20, 20), -1)


def draw_truck(image, x, y):
    """Draw a truck-like shape"""
    # Cabin
    cv2.rectangle(image, (x, y + 10), (x + 40, y + 60), (100, 100, 150), -1)
    
    # Cargo area
    cv2.rectangle(image, (x + 35, y), (x + 120, y + 60), (150, 150, 180), -1)
    cv2.rectangle(image, (x + 35, y), (x + 120, y + 60), (255, 255, 255), 2)
    
    # Wheels
    cv2.circle(image, (x + 25, y + 60), 11, (20, 20, 20), -1)
    cv2.circle(image, (x + 100, y + 60), 11, (20, 20, 20), -1)


def main():
    download_sample_images()


if __name__ == "__main__":
    main()
