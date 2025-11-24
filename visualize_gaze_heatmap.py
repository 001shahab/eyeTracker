"""
Gaze Data Heatmap Visualization
Creates a heatmap visualization of gaze tracking data
(c) Prof. Shahab Anbarjafari - 3S Holding
"""

import json
import sys
import numpy as np
import cv2
from datetime import datetime


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


def create_heatmap(data, output_filename='results/gaze_heatmap.png'):
    """Create a heatmap visualization of gaze data."""
    if not data or 'gaze_data' not in data:
        print("Error: No gaze data found!")
        return
    
    gaze_points = data['gaze_data']
    
    if len(gaze_points) == 0:
        print("Error: No gaze points recorded!")
        return
    
    # Get monitor dimensions
    monitor = gaze_points[0].get('monitor')
    if not monitor:
        print("Error: No monitor information found!")
        return
    
    width = monitor['width']
    height = monitor['height']
    
    print(f"Creating heatmap for {width}x{height} monitor...")
    
    # Check if calibration data is available
    has_calibration = data.get('calibration_points') is not None
    if has_calibration:
        print(f"  ✓ Calibrated data detected ({len(data['calibration_points'])} calibration points)")
    else:
        print(f"  ℹ No calibration data (basic eye tracking)")
    
    # Create heatmap array (scaled down for performance)
    scale = 2  # Scale factor (higher = lower resolution but faster)
    heatmap_width = width // scale
    heatmap_height = height // scale
    heatmap = np.zeros((heatmap_height, heatmap_width), dtype=np.float32)
    
    # Populate heatmap
    valid_points = 0
    for point in gaze_points:
        if point['screen_x'] is not None and point['screen_y'] is not None:
            x = int((point['screen_x'] - monitor['x']) / scale)
            y = int((point['screen_y'] - monitor['y']) / scale)
            
            if 0 <= x < heatmap_width and 0 <= y < heatmap_height:
                heatmap[y, x] += 1
                valid_points += 1
    
    print(f"Processed {valid_points} valid gaze points out of {len(gaze_points)} total")
    
    if valid_points == 0:
        print("Error: No valid points to visualize!")
        return
    
    # Apply Gaussian blur for smoother heatmap
    heatmap = cv2.GaussianBlur(heatmap, (51, 51), 0)
    
    # Normalize to 0-255 range
    if heatmap.max() > 0:
        heatmap = (heatmap / heatmap.max() * 255).astype(np.uint8)
    
    # Apply color map (hot = red/yellow for high attention)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Scale back to original resolution
    heatmap_color = cv2.resize(heatmap_color, (width, height), 
                               interpolation=cv2.INTER_LINEAR)
    
    # Create a white background
    background = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Blend heatmap with background
    alpha = 0.7  # Transparency of heatmap
    overlay = cv2.addWeighted(background, 1 - alpha, heatmap_color, alpha, 0)
    
    # Add title and info
    font = cv2.FONT_HERSHEY_SIMPLEX
    title = "Gaze Tracking Heatmap"
    info = f"Total Samples: {valid_points} | Session: {data.get('session_start', 'N/A')}"
    copyright_text = "(c) Prof. Shahab Anbarjafari - 3S Holding"
    
    # Add black background for text
    cv2.rectangle(overlay, (0, 0), (width, 100), (0, 0, 0), -1)
    cv2.rectangle(overlay, (0, height - 40), (width, height), (0, 0, 0), -1)
    
    # Add text
    cv2.putText(overlay, title, (20, 40), font, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(overlay, info, (20, 75), font, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(overlay, copyright_text, (20, height - 10), font, 0.5, 
                (150, 150, 150), 1, cv2.LINE_AA)
    
    # Add color scale legend
    legend_height = 200
    legend_width = 30
    legend = np.zeros((legend_height, legend_width), dtype=np.uint8)
    for i in range(legend_height):
        legend[i, :] = int((legend_height - i) / legend_height * 255)
    
    legend_color = cv2.applyColorMap(legend, cv2.COLORMAP_JET)
    
    # Position legend in top right
    legend_x = width - legend_width - 20
    legend_y = 120
    
    # Add legend background
    cv2.rectangle(overlay, 
                 (legend_x - 5, legend_y - 5),
                 (legend_x + legend_width + 5, legend_y + legend_height + 5),
                 (0, 0, 0), -1)
    
    # Add legend
    overlay[legend_y:legend_y + legend_height, legend_x:legend_x + legend_width] = legend_color
    
    # Add legend labels
    cv2.putText(overlay, "High", (legend_x - 50, legend_y + 15), 
                font, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(overlay, "Low", (legend_x - 50, legend_y + legend_height), 
                font, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    
    # Draw calibration points if available
    if data.get('calibration_points'):
        cal_points = data['calibration_points']
        for idx, (cx, cy) in enumerate(cal_points):
            # Convert to monitor-relative coordinates
            cx_rel = cx - monitor['x']
            cy_rel = cy - monitor['y']
            
            if 0 <= cx_rel < width and 0 <= cy_rel < height:
                # Draw calibration point marker
                cv2.circle(overlay, (cx_rel, cy_rel), 8, (255, 255, 255), 2)
                cv2.circle(overlay, (cx_rel, cy_rel), 6, (0, 0, 0), 2)
                cv2.circle(overlay, (cx_rel, cy_rel), 3, (0, 255, 255), -1)
                
                # Add number label
                cv2.putText(overlay, str(idx + 1), (cx_rel + 12, cy_rel + 5),
                           font, 0.4, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(overlay, str(idx + 1), (cx_rel + 12, cy_rel + 5),
                           font, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
    
    # Save heatmap
    cv2.imwrite(output_filename, overlay)
    print(f"\nHeatmap saved to: {output_filename}")
    
    # Display heatmap
    print("Displaying heatmap... (Press any key to close)")
    
    # Scale down for display if too large
    display_scale = 1.0
    if width > 1920 or height > 1080:
        display_scale = min(1920 / width, 1080 / height)
    
    if display_scale < 1.0:
        display_width = int(width * display_scale)
        display_height = int(height * display_scale)
        display_img = cv2.resize(overlay, (display_width, display_height))
    else:
        display_img = overlay
    
    cv2.imshow("Gaze Tracking Heatmap", display_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return overlay


def create_trajectory_visualization(data, output_filename='results/gaze_trajectory.png'):
    """Create a trajectory visualization showing gaze path over time."""
    if not data or 'gaze_data' not in data:
        print("Error: No gaze data found!")
        return
    
    gaze_points = data['gaze_data']
    
    if len(gaze_points) == 0:
        print("Error: No gaze points recorded!")
        return
    
    # Get monitor dimensions
    monitor = gaze_points[0].get('monitor')
    if not monitor:
        print("Error: No monitor information found!")
        return
    
    width = monitor['width']
    height = monitor['height']
    
    print(f"\nCreating trajectory visualization for {width}x{height} monitor...")
    
    # Check if calibration data is available
    has_calibration = data.get('calibration_points') is not None
    if has_calibration:
        print(f"  ✓ Will overlay {len(data['calibration_points'])} calibration points")
    
    # Create blank canvas
    canvas = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Extract valid points
    points = []
    for point in gaze_points:
        if point['screen_x'] is not None and point['screen_y'] is not None:
            x = int(point['screen_x'] - monitor['x'])
            y = int(point['screen_y'] - monitor['y'])
            
            if 0 <= x < width and 0 <= y < height:
                points.append((x, y))
    
    if len(points) < 2:
        print("Error: Not enough valid points to create trajectory!")
        return
    
    print(f"Drawing trajectory with {len(points)} points...")
    
    # Draw trajectory with gradient color (blue -> red over time)
    for i in range(len(points) - 1):
        # Calculate color based on time progression
        progress = i / len(points)
        color = (
            int(255 * progress),        # Red increases
            int(100 * (1 - progress)),  # Green
            int(255 * (1 - progress))   # Blue decreases
        )
        
        # Draw line
        cv2.line(canvas, points[i], points[i + 1], color, 2, cv2.LINE_AA)
        
        # Draw point every 10th sample
        if i % 10 == 0:
            cv2.circle(canvas, points[i], 3, color, -1)
    
    # Mark start and end
    cv2.circle(canvas, points[0], 10, (0, 255, 0), -1)  # Green = start
    cv2.circle(canvas, points[0], 12, (0, 0, 0), 2)
    cv2.circle(canvas, points[-1], 10, (0, 0, 255), -1)  # Red = end
    cv2.circle(canvas, points[-1], 12, (0, 0, 0), 2)
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, "START", (points[0][0] + 15, points[0][1]), 
                font, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(canvas, "END", (points[-1][0] + 15, points[-1][1]), 
                font, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    
    # Draw calibration points if available (faint markers)
    if data.get('calibration_points'):
        cal_points = data['calibration_points']
        for idx, (cx, cy) in enumerate(cal_points):
            cx_rel = cx - monitor['x']
            cy_rel = cy - monitor['y']
            
            if 0 <= cx_rel < width and 0 <= cy_rel < height:
                # Draw faint calibration point marker
                cv2.circle(canvas, (cx_rel, cy_rel), 6, (200, 200, 200), 1)
                cv2.circle(canvas, (cx_rel, cy_rel), 2, (150, 150, 150), -1)
    
    # Add title and info
    cv2.rectangle(canvas, (0, 0), (width, 80), (0, 0, 0), -1)
    cv2.rectangle(canvas, (0, height - 40), (width, height), (0, 0, 0), -1)
    
    title = "Gaze Trajectory Visualization"
    info = f"Points: {len(points)} | Blue (start) -> Red (end)"
    copyright_text = "(c) Prof. Shahab Anbarjafari - 3S Holding"
    
    cv2.putText(canvas, title, (20, 35), font, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, info, (20, 60), font, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(canvas, copyright_text, (20, height - 10), font, 0.5, 
                (150, 150, 150), 1, cv2.LINE_AA)
    
    # Save trajectory
    cv2.imwrite(output_filename, canvas)
    print(f"Trajectory saved to: {output_filename}")
    
    # Display trajectory
    print("Displaying trajectory... (Press any key to close)")
    
    # Scale for display
    display_scale = 1.0
    if width > 1920 or height > 1080:
        display_scale = min(1920 / width, 1080 / height)
    
    if display_scale < 1.0:
        display_width = int(width * display_scale)
        display_height = int(height * display_scale)
        display_img = cv2.resize(canvas, (display_width, display_height))
    else:
        display_img = canvas
    
    cv2.imshow("Gaze Trajectory", display_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    """Main entry point."""
    print("="*70)
    print("Gaze Data Visualization Tool")
    print("(c) Prof. Shahab Anbarjafari - 3S Holding")
    print("="*70)
    print()
    
    if len(sys.argv) < 2:
        print("Usage: python visualize_gaze_heatmap.py <gaze_tracking_file.json>")
        print()
        print("Example: python visualize_gaze_heatmap.py results/gaze_tracking_20231124_143022.json")
        print()
        print("This will create:")
        print("  - results/gaze_heatmap.png: Heatmap showing attention density")
        print("  - results/gaze_trajectory.png: Path of eye movement over time")
        sys.exit(1)
    
    filename = sys.argv[1]
    data = load_gaze_data(filename)
    
    if data:
        # Create heatmap
        create_heatmap(data, 'results/gaze_heatmap.png')
        
        # Create trajectory
        create_trajectory_visualization(data, 'results/gaze_trajectory.png')
        
        print("\n" + "="*70)
        print("Visualization Complete!")
        print("="*70)


if __name__ == "__main__":
    main()

