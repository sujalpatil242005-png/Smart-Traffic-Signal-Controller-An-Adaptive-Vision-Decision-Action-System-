"""
Smart Traffic Signal Controller - Main Application
Vision-Decision-Action Loop Implementation

This system:
1. Loads 4 road images (North, South, East, West)
2. Detects vehicles using YOLOv8
3. Calculates weighted density scores
4. Allocates green times adaptively with constraints
5. Displays a 2x2 dashboard with HUD overlays
"""

import cv2
import os
import sys
from pathlib import Path
from traffic_controller import TrafficController
import config


def check_images_exist(image_dir: Path) -> bool:
    """Check if all 4 required images exist"""
    required_files = ['north.jpg', 'south.jpg', 'east.jpg', 'west.jpg']
    
    for filename in required_files:
        filepath = image_dir / filename
        if not filepath.exists():
            # Try alternative extensions
            alternatives = ['.png', '.jpeg', '.JPG', '.PNG']
            found = False
            for ext in alternatives:
                alt_path = image_dir / filename.replace('.jpg', ext)
                if alt_path.exists():
                    found = True
                    break
            if not found:
                return False
    return True


def get_image_paths(image_dir: Path) -> list:
    """Get paths to the 4 road images with flexible extensions"""
    roads = ['north', 'south', 'east', 'west']
    extensions = ['.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG']
    
    image_paths = []
    for road in roads:
        found = False
        for ext in extensions:
            filepath = image_dir / f"{road}{ext}"
            if filepath.exists():
                image_paths.append(str(filepath))
                found = True
                break
        
        if not found:
            raise FileNotFoundError(
                f"Could not find image for {road} road. "
                f"Please ensure {road}.jpg (or .png) exists in {image_dir}"
            )
    
    return image_paths


def main():
    """Main execution function"""
    # Check for demo mode flag
    demo_mode = '--demo' in sys.argv or os.getenv('DEMO_MODE') == '1'
    
    print("\n" + "="*70)
    print(" "*15 + "SMART TRAFFIC SIGNAL CONTROLLER")
    if demo_mode:
        print(" "*18 + "🎭 DEMO MODE - Using JSON Data")
    else:
        print(" "*20 + "Vision-Decision-Action System")
    print("="*70 + "\n")
    
    # Setup paths
    project_dir = Path(__file__).parent

    if demo_mode:
        image_dir = project_dir / "images" / "synthetic"
        print(f"[MODE] Demo Mode: Using synthetic images from {image_dir}")
    else:
        image_dir = project_dir / "images" / "real"
        print(f"[MODE] AI Mode: Using real images from {image_dir}")
    
    # Check if images directory exists
    if not image_dir.exists():
        print(f"[ERROR] Images directory not found: {image_dir}")
        print("Please create the 'images' folder and add 4 test images:")
        print("  - north.jpg")
        print("  - south.jpg")
        print("  - east.jpg")
        print("  - west.jpg")
        return
    
    # Check if images exist
    if not check_images_exist(image_dir):
        if not demo_mode:
            print(f"[ERROR] Missing traffic images in: {image_dir}")
            print("\nRequired images:")
            print("  1. north.jpg - View from North road")
            print("  2. south.jpg - View from South road")
            print("  3. east.jpg  - View from East road")
            print("  4. west.jpg  - View from West road")
            print("\nTip: You can use the generate_images.py script to create sample images")
            return
        else:
            print("[WARNING] Missing some images, but running in demo mode...")
    
    # Get image paths
    try:
        image_paths = get_image_paths(image_dir)
    except FileNotFoundError as e:
        if demo_mode:
            print(f"[WARNING] {e}")
            print("Using existing images for visualization only...")
            # Create dummy paths - images will only be used for visualization in demo mode
            image_paths = [str(image_dir / f"{road}.jpg") for road in ['north', 'south', 'east', 'west']]
        else:
            print(f"[ERROR] {e}")
            return
    
    # Initialize Traffic Controller
    controller = TrafficController(demo_mode=demo_mode)
    
    # Process the intersection
    dashboard, analytics = controller.process_intersection(image_paths)
    
    # Display results
    print("\n📊 ANALYTICS SUMMARY")
    print("-" * 50)
    print(f"Total Cycle Time: {config.TOTAL_CYCLE_TIME}s")
    print(f"Total Vehicles: {analytics['total_vehicles']}")
    print(f"Avg Vehicles per Road: {analytics['total_vehicles'] / 4:.1f}")
    
    # Display the dashboard
    print("\n[DISPLAY] Showing traffic dashboard...")
    print("Press any key to close the window")
    
    cv2.imshow("Smart Traffic Signal Controller - Dashboard", dashboard)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Optionally save the dashboard
    if demo_mode:
        print("\n💡 TIP: Run without --demo flag to use real YOLO detection")
    output_path = project_dir / "traffic_dashboard.jpg"
    cv2.imwrite(str(output_path), dashboard)
    print(f"\n✅ Dashboard saved to: {output_path}")
    
    print("\n" + "="*70)
    print(" "*25 + "CYCLE COMPLETE")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
