"""
Gaze Data Analysis Script
Analyzes the JSON output from the eye tracking service
(c) Prof. Shahab Anbarjafari - 3S Holding
"""

import json
import sys
import numpy as np
from datetime import datetime
from collections import Counter


def load_gaze_data(filename):
    """Load gaze data from JSON file."""
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found!")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in '{filename}'!")
        return None


def analyze_gaze_data(data):
    """Perform comprehensive analysis of gaze data."""
    if not data or 'gaze_data' not in data:
        print("Error: No gaze data found!")
        return
    
    gaze_points = data['gaze_data']
    
    if len(gaze_points) == 0:
        print("Error: No gaze points recorded!")
        return
    
    print("="*70)
    print("GAZE TRACKING DATA ANALYSIS")
    print("(c) Prof. Shahab Anbarjafari - 3S Holding")
    print("="*70)
    print()
    
    # Check data format version
    has_calibration = 'calibration_points' in data
    has_eye_features = gaze_points[0].get('eye_features') is not None if len(gaze_points) > 0 else False
    
    if has_calibration:
        print("📊 Data Format: Calibrated Gaze Tracking (with personal calibration)")
    else:
        print("📊 Data Format: Basic Eye Tracking (without calibration)")
    print()
    
    # Session information
    print("SESSION INFORMATION:")
    print("-" * 70)
    if data.get('session_start'):
        start_time = datetime.fromisoformat(data['session_start'])
        print(f"  Session Start: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    if len(gaze_points) > 0:
        end_time = datetime.fromisoformat(gaze_points[-1]['timestamp'])
        print(f"  Session End:   {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        if data.get('session_start'):
            duration = end_time - start_time
            print(f"  Duration:      {duration.total_seconds():.1f} seconds")
    
    print(f"  Total Samples: {data['total_samples']}")
    
    if len(gaze_points) > 1 and data.get('session_start'):
        duration = (end_time - start_time).total_seconds()
        if duration > 0:
            print(f"  Sampling Rate: {data['total_samples'] / duration:.1f} Hz")
    print()
    
    # Monitor information
    if gaze_points[0].get('monitor'):
        monitor = gaze_points[0]['monitor']
        print("MONITOR INFORMATION:")
        print("-" * 70)
        print(f"  Resolution: {monitor['width']}x{monitor['height']}")
        print(f"  Position:   ({monitor['x']}, {monitor['y']})")
        print()
    
    # Extract valid screen coordinates
    screen_coords = []
    for point in gaze_points:
        if point['screen_x'] is not None and point['screen_y'] is not None:
            screen_coords.append([point['screen_x'], point['screen_y']])
    
    if len(screen_coords) == 0:
        print("Warning: No valid screen coordinates found!")
        return
    
    screen_coords = np.array(screen_coords)
    
    # Statistical analysis
    print("SCREEN COORDINATE STATISTICS:")
    print("-" * 70)
    print(f"  Valid Samples: {len(screen_coords)} / {len(gaze_points)}")
    print(f"  Coverage:      {100 * len(screen_coords) / len(gaze_points):.1f}%")
    print()
    
    print("  X Coordinates (Horizontal):")
    print(f"    Min:      {screen_coords[:, 0].min():.0f} px")
    print(f"    Max:      {screen_coords[:, 0].max():.0f} px")
    print(f"    Mean:     {screen_coords[:, 0].mean():.0f} px")
    print(f"    Median:   {np.median(screen_coords[:, 0]):.0f} px")
    print(f"    Std Dev:  {screen_coords[:, 0].std():.0f} px")
    print()
    
    print("  Y Coordinates (Vertical):")
    print(f"    Min:      {screen_coords[:, 1].min():.0f} px")
    print(f"    Max:      {screen_coords[:, 1].max():.0f} px")
    print(f"    Mean:     {screen_coords[:, 1].mean():.0f} px")
    print(f"    Median:   {np.median(screen_coords[:, 1]):.0f} px")
    print(f"    Std Dev:  {screen_coords[:, 1].std():.0f} px")
    print()
    
    # Screen region analysis (divide screen into 9 regions)
    monitor = gaze_points[0]['monitor']
    width = monitor['width']
    height = monitor['height']
    
    regions = {
        'Top-Left': 0, 'Top-Center': 0, 'Top-Right': 0,
        'Middle-Left': 0, 'Middle-Center': 0, 'Middle-Right': 0,
        'Bottom-Left': 0, 'Bottom-Center': 0, 'Bottom-Right': 0
    }
    
    for x, y in screen_coords:
        # Normalize to monitor coordinates
        x_norm = x - monitor['x']
        y_norm = y - monitor['y']
        
        # Determine region
        if x_norm < width / 3:
            col = 'Left'
        elif x_norm < 2 * width / 3:
            col = 'Center'
        else:
            col = 'Right'
        
        if y_norm < height / 3:
            row = 'Top'
        elif y_norm < 2 * height / 3:
            row = 'Middle'
        else:
            row = 'Bottom'
        
        region = f"{row}-{col}"
        regions[region] += 1
    
    print("SCREEN REGION ANALYSIS:")
    print("-" * 70)
    print("  (Screen divided into 3x3 grid)")
    print()
    
    total = len(screen_coords)
    for region, count in regions.items():
        percentage = 100 * count / total
        bar = '█' * int(percentage / 2)
        print(f"  {region:15s}: {count:5d} samples ({percentage:5.1f}%) {bar}")
    print()
    
    # Calibration information
    if data.get('calibration_points'):
        cal_points = data['calibration_points']
        print("CALIBRATION INFORMATION:")
        print("-" * 70)
        print(f"  Calibration Points: {len(cal_points)}")
        print(f"  Calibration Grid:")
        for idx, (x, y) in enumerate(cal_points[:9]):  # Show up to 9 points
            if idx % 3 == 0:
                print(f"    ", end="")
            print(f"({x:4d},{y:4d})  ", end="")
            if (idx + 1) % 3 == 0:
                print()
        print()
    
    # Eye features analysis (if available)
    eye_features = []
    for point in gaze_points:
        if point.get('eye_features') is not None:
            eye_features.append(point['eye_features'])
    
    if len(eye_features) > 0:
        eye_features = np.array(eye_features)
        print("EYE FEATURES ANALYSIS:")
        print("-" * 70)
        print("  Feature Statistics (9D feature vector):")
        print()
        
        feature_names = [
            "Avg Iris Offset X",
            "Avg Iris Offset Y",
            "Left Iris Offset X",
            "Left Iris Offset Y",
            "Right Iris Offset X",
            "Right Iris Offset Y",
            "Head Pose X",
            "Head Pose Y",
            "Head Tilt"
        ]
        
        for i, name in enumerate(feature_names):
            if i < eye_features.shape[1]:
                feat_data = eye_features[:, i]
                print(f"  {name:20s}:")
                print(f"    Range: [{feat_data.min():7.4f}, {feat_data.max():7.4f}]")
                print(f"    Mean:  {feat_data.mean():7.4f}  Std: {feat_data.std():7.4f}")
        print()
    
    # Most fixated point
    print("GAZE FIXATION ANALYSIS:")
    print("-" * 70)
    
    # Find the most common gaze region (within 50px radius)
    from scipy import ndimage
    if len(screen_coords) > 0:
        # Create heatmap
        heatmap = np.zeros((height // 10, width // 10))
        for x, y in screen_coords:
            x_norm = int((x - monitor['x']) / 10)
            y_norm = int((y - monitor['y']) / 10)
            if 0 <= x_norm < width // 10 and 0 <= y_norm < height // 10:
                heatmap[y_norm, x_norm] += 1
        
        # Find hotspot
        max_idx = np.unravel_index(heatmap.argmax(), heatmap.shape)
        hotspot_x = (max_idx[1] * 10) + monitor['x']
        hotspot_y = (max_idx[0] * 10) + monitor['y']
        
        print(f"  Primary Fixation Point: ({hotspot_x:.0f}, {hotspot_y:.0f})")
        print(f"  Fixation Density: {heatmap.max():.0f} samples in 10x10px region")
        print()
    
    print("="*70)
    print("Analysis Complete!")
    print("="*70)


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python analyze_gaze_data.py <gaze_tracking_file.json>")
        print()
        print("Example: python analyze_gaze_data.py results/gaze_tracking_20231124_143022.json")
        sys.exit(1)
    
    filename = sys.argv[1]
    data = load_gaze_data(filename)
    
    if data:
        analyze_gaze_data(data)


if __name__ == "__main__":
    main()

