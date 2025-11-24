"""
Eye Gaze Tracking Service with Calibration and Multi-Monitor Support
Tracks where on screen the person is looking (not just eye movement)
(c) Prof. Shahab Anbarjafari - 3S Holding
"""

import cv2
import mediapipe as mp
import numpy as np
import json
import time
from datetime import datetime
from screeninfo import get_monitors
import sys
from scipy.interpolate import Rbf


class GazeTrackingService:
    def __init__(self):
        """Initialize the gaze tracking service with MediaPipe and OpenCV."""
        # MediaPipe Face Mesh for accurate eye tracking
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # OpenCV video capture
        self.cap = None
        
        # Monitor information
        self.monitors = list(get_monitors())
        self.selected_monitor = None
        
        # Gaze tracking data
        self.gaze_history = []
        self.json_filename = f"results/gaze_tracking_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Calibration data and models
        self.calibration_points = []  # Screen points for calibration
        self.calibration_features = []  # Eye features at those points
        self.is_calibrated = False
        self.gaze_model_x = None
        self.gaze_model_y = None
        
        # Eye landmarks indices for MediaPipe Face Mesh
        self.LEFT_EYE_INDICES = [33, 133, 160, 159, 158, 157, 173, 246]
        self.RIGHT_EYE_INDICES = [362, 263, 387, 386, 385, 384, 398, 466]
        self.LEFT_IRIS_INDICES = [468, 469, 470, 471, 472]
        self.RIGHT_IRIS_INDICES = [473, 474, 475, 476, 477]
        
    def select_monitor(self):
        """Allow user to select which monitor to track."""
        if len(self.monitors) == 0:
            print("Error: No monitors detected!")
            sys.exit(1)
        elif len(self.monitors) == 1:
            self.selected_monitor = self.monitors[0]
            print(f"Using monitor: {self.selected_monitor.name if hasattr(self.selected_monitor, 'name') else 'Display'} ({self.selected_monitor.width}x{self.selected_monitor.height})")
            return
        
        print("\n" + "="*60)
        print("Multiple monitors detected. Please select one:")
        print("="*60)
        for idx, monitor in enumerate(self.monitors):
            print(f"[{idx}] Monitor: {monitor.name if hasattr(monitor, 'name') else f'Monitor {idx}'}")
            print(f"    Resolution: {monitor.width}x{monitor.height}")
            print(f"    Position: ({monitor.x}, {monitor.y})")
            if hasattr(monitor, 'is_primary') and monitor.is_primary:
                print("    [PRIMARY]")
            print()
        
        while True:
            try:
                choice = input(f"Select monitor [0-{len(self.monitors)-1}]: ")
                choice = int(choice)
                if 0 <= choice < len(self.monitors):
                    self.selected_monitor = self.monitors[choice]
                    print(f"\nSelected: Monitor {choice}")
                    break
                else:
                    print(f"Please enter a number between 0 and {len(self.monitors)-1}")
            except ValueError:
                print("Please enter a valid number")
            except KeyboardInterrupt:
                print("\nExiting...")
                sys.exit(0)
    
    def initialize_camera(self):
        """Initialize the webcam."""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open webcam!")
            sys.exit(1)
        
        # Set camera resolution for better accuracy
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print("Camera initialized successfully!")
    
    def get_eye_features(self, landmarks, frame_shape):
        """
        Extract eye features for gaze estimation.
        Returns normalized features that correlate with gaze direction.
        """
        h, w = frame_shape[:2]
        
        # Get left eye center and iris
        left_eye_pts = []
        for idx in self.LEFT_EYE_INDICES:
            if idx < len(landmarks):
                lm = landmarks[idx]
                left_eye_pts.append([lm.x, lm.y])
        
        left_iris_pts = []
        for idx in self.LEFT_IRIS_INDICES:
            if idx < len(landmarks):
                lm = landmarks[idx]
                left_iris_pts.append([lm.x, lm.y])
        
        # Get right eye center and iris
        right_eye_pts = []
        for idx in self.RIGHT_EYE_INDICES:
            if idx < len(landmarks):
                lm = landmarks[idx]
                right_eye_pts.append([lm.x, lm.y])
        
        right_iris_pts = []
        for idx in self.RIGHT_IRIS_INDICES:
            if idx < len(landmarks):
                lm = landmarks[idx]
                right_iris_pts.append([lm.x, lm.y])
        
        if len(left_eye_pts) == 0 or len(left_iris_pts) == 0 or \
           len(right_eye_pts) == 0 or len(right_iris_pts) == 0:
            return None
        
        # Calculate centers
        left_eye_center = np.mean(left_eye_pts, axis=0)
        left_iris_center = np.mean(left_iris_pts, axis=0)
        right_eye_center = np.mean(right_eye_pts, axis=0)
        right_iris_center = np.mean(right_iris_pts, axis=0)
        
        # Calculate iris offset from eye center (normalized)
        left_offset = left_iris_center - left_eye_center
        right_offset = right_iris_center - right_eye_center
        
        # Average both eyes for robustness
        avg_offset = (left_offset + right_offset) / 2
        
        # Get head pose indicators (face landmarks for 3D orientation)
        nose_tip = landmarks[1]  # Nose tip
        chin = landmarks[152]  # Chin
        
        # Feature vector: [iris_x, iris_y, head_orientation_indicators]
        features = np.array([
            avg_offset[0],  # Horizontal iris offset
            avg_offset[1],  # Vertical iris offset
            left_offset[0],  # Left eye horizontal
            left_offset[1],  # Left eye vertical
            right_offset[0],  # Right eye horizontal
            right_offset[1],  # Right eye vertical
            nose_tip.x,  # Head pose x
            nose_tip.y,  # Head pose y
            chin.y - nose_tip.y,  # Vertical head tilt indicator
        ])
        
        return features
    
    def run_calibration(self):
        """
        Run calibration sequence where user looks at specific points on screen.
        """
        print("\n" + "="*60)
        print("CALIBRATION PHASE")
        print("="*60)
        print("You will see red dots appear on your screen.")
        print("Look directly at each dot until it turns GREEN.")
        print("Try to keep your head still during calibration.")
        print("\nPress any key to start calibration...")
        print("="*60)
        
        # Wait for user
        while True:
            ret, frame = self.cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
            cv2.putText(frame, "Press any key to start calibration", 
                       (50, 360), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Gaze Tracking - Calibration", frame)
            if cv2.waitKey(1) != -1:
                break
        
        # Define calibration points (9-point calibration grid)
        width = self.selected_monitor.width
        height = self.selected_monitor.height
        offset_x = self.selected_monitor.x
        offset_y = self.selected_monitor.y
        
        # Create calibration grid: 3x3 with margins
        margin_x = width // 6
        margin_y = height // 6
        
        cal_points = [
            (offset_x + margin_x, offset_y + margin_y),  # Top-left
            (offset_x + width // 2, offset_y + margin_y),  # Top-center
            (offset_x + width - margin_x, offset_y + margin_y),  # Top-right
            (offset_x + margin_x, offset_y + height // 2),  # Middle-left
            (offset_x + width // 2, offset_y + height // 2),  # Center
            (offset_x + width - margin_x, offset_y + height // 2),  # Middle-right
            (offset_x + margin_x, offset_y + height - margin_y),  # Bottom-left
            (offset_x + width // 2, offset_y + height - margin_y),  # Bottom-center
            (offset_x + width - margin_x, offset_y + height - margin_y),  # Bottom-right
        ]
        
        # Create calibration window on selected monitor
        cv2.namedWindow("Calibration Target", cv2.WINDOW_NORMAL)
        cv2.moveWindow("Calibration Target", offset_x, offset_y)
        cv2.setWindowProperty("Calibration Target", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        self.calibration_points = []
        self.calibration_features = []
        
        for idx, (px, py) in enumerate(cal_points):
            print(f"Calibration point {idx + 1}/{len(cal_points)}: ({px}, {py})")
            
            # Show calibration target
            target_img = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Draw dot position (relative to monitor)
            dot_x = px - offset_x
            dot_y = py - offset_y
            
            # Collect samples for this point
            samples = []
            sample_count = 0
            required_samples = 30  # Collect 30 frames per point
            
            start_time = time.time()
            
            while sample_count < required_samples:
                # Clear target
                target_img = np.zeros((height, width, 3), dtype=np.uint8)
                
                # Draw progress ring
                progress = sample_count / required_samples
                cv2.circle(target_img, (dot_x, dot_y), 40, (50, 50, 50), 2)
                angle = int(360 * progress)
                cv2.ellipse(target_img, (dot_x, dot_y), (40, 40), 0, -90, -90 + angle, (0, 255, 0), 3)
                
                # Draw calibration dot (red -> green when complete)
                color = (0, int(255 * progress), int(255 * (1 - progress)))
                cv2.circle(target_img, (dot_x, dot_y), 20, color, -1)
                
                # Add instructions
                cv2.putText(target_img, f"Look at the dot ({idx + 1}/{len(cal_points)})",
                           (width // 2 - 200, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                cv2.imshow("Calibration Target", target_img)
                cv2.waitKey(1)
                
                # Capture eye features
                ret, frame = self.cap.read()
                if not ret:
                    continue
                
                frame = cv2.flip(frame, 1)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.face_mesh.process(frame_rgb)
                
                if results.multi_face_landmarks:
                    landmarks = results.multi_face_landmarks[0].landmark
                    features = self.get_eye_features(landmarks, frame.shape)
                    
                    if features is not None:
                        samples.append(features)
                        sample_count += 1
                
                # Show camera feed in small window
                small_frame = cv2.resize(frame, (320, 240))
                cv2.imshow("Gaze Tracking - Calibration", small_frame)
                
                # Timeout after 10 seconds
                if time.time() - start_time > 10:
                    print(f"  Warning: Timeout for point {idx + 1}")
                    break
            
            if len(samples) > 0:
                # Average samples for this calibration point
                avg_features = np.mean(samples, axis=0)
                self.calibration_features.append(avg_features)
                self.calibration_points.append([px, py])
                print(f"  ✓ Collected {len(samples)} samples")
            else:
                print(f"  ✗ Failed to collect samples for point {idx + 1}")
        
        cv2.destroyWindow("Calibration Target")
        
        # Build gaze estimation model
        if len(self.calibration_points) >= 5:
            self._build_gaze_model()
            print("\n✓ Calibration complete!")
            print(f"  Calibrated with {len(self.calibration_points)} points")
            self.is_calibrated = True
        else:
            print("\n✗ Calibration failed: insufficient calibration points")
            print(f"  Only collected {len(self.calibration_points)} points (need at least 5)")
            self.is_calibrated = False
    
    def _build_gaze_model(self):
        """Build RBF interpolation model for gaze estimation."""
        cal_features = np.array(self.calibration_features)
        cal_points = np.array(self.calibration_points)
        
        # Create separate models for X and Y coordinates
        # Using Radial Basis Function interpolation for smooth mapping
        try:
            self.gaze_model_x = Rbf(
                cal_features[:, 0],  # avg iris x
                cal_features[:, 1],  # avg iris y
                cal_features[:, 2],  # left iris x
                cal_features[:, 3],  # left iris y
                cal_features[:, 4],  # right iris x
                cal_features[:, 5],  # right iris y
                cal_features[:, 6],  # nose x
                cal_features[:, 7],  # nose y
                cal_features[:, 8],  # head tilt
                cal_points[:, 0],    # screen x
                function='multiquadric',
                smooth=0.1
            )
            
            self.gaze_model_y = Rbf(
                cal_features[:, 0],
                cal_features[:, 1],
                cal_features[:, 2],
                cal_features[:, 3],
                cal_features[:, 4],
                cal_features[:, 5],
                cal_features[:, 6],
                cal_features[:, 7],
                cal_features[:, 8],
                cal_points[:, 1],    # screen y
                function='multiquadric',
                smooth=0.1
            )
            
            print("  ✓ Gaze estimation model built successfully")
        except Exception as e:
            print(f"  ✗ Error building gaze model: {e}")
            self.is_calibrated = False
    
    def estimate_gaze_point(self, features):
        """Estimate screen coordinates from eye features using calibrated model."""
        if not self.is_calibrated or self.gaze_model_x is None:
            return None, None
        
        try:
            screen_x = self.gaze_model_x(
                features[0], features[1], features[2], features[3],
                features[4], features[5], features[6], features[7], features[8]
            )
            screen_y = self.gaze_model_y(
                features[0], features[1], features[2], features[3],
                features[4], features[5], features[6], features[7], features[8]
            )
            
            # Clamp to monitor bounds
            screen_x = max(self.selected_monitor.x,
                          min(screen_x, self.selected_monitor.x + self.selected_monitor.width - 1))
            screen_y = max(self.selected_monitor.y,
                          min(screen_y, self.selected_monitor.y + self.selected_monitor.height - 1))
            
            return int(screen_x), int(screen_y)
        except Exception as e:
            return None, None
    
    def save_gaze_data(self, screen_x, screen_y, features):
        """Save gaze data to JSON file."""
        data_point = {
            'timestamp': datetime.now().isoformat(),
            'screen_x': int(screen_x) if screen_x is not None else None,
            'screen_y': int(screen_y) if screen_y is not None else None,
            'eye_features': features.tolist() if features is not None else None,
            'monitor': {
                'width': self.selected_monitor.width,
                'height': self.selected_monitor.height,
                'x': self.selected_monitor.x,
                'y': self.selected_monitor.y
            }
        }
        
        self.gaze_history.append(data_point)
        
        # Save to file (overwrite each time for dynamic updates)
        with open(self.json_filename, 'w') as f:
            json.dump({
                'session_start': self.gaze_history[0]['timestamp'] if self.gaze_history else None,
                'total_samples': len(self.gaze_history),
                'calibration_points': [[int(p[0]), int(p[1])] for p in self.calibration_points],
                'gaze_data': self.gaze_history
            }, f, indent=2)
    
    def add_text_overlay(self, frame, screen_x, screen_y):
        """Add coordinate display and copyright notice to frame."""
        h, w = frame.shape[:2]
        
        # Bottom right: Screen coordinates (medium font)
        if screen_x is not None and screen_y is not None:
            coord_text = f"Gaze: ({screen_x}, {screen_y})"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            
            # Get text size for background
            (text_width, text_height), baseline = cv2.getTextSize(
                coord_text, font, font_scale, thickness
            )
            
            # Position: bottom right with padding
            padding = 10
            x = w - text_width - padding
            y = h - padding
            
            # Draw semi-transparent background
            overlay = frame.copy()
            cv2.rectangle(overlay, 
                         (x - 5, y - text_height - 5),
                         (x + text_width + 5, y + baseline + 5),
                         (0, 0, 0), -1)
            frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
            
            # Draw text
            cv2.putText(frame, coord_text, (x, y), font, font_scale, 
                       (0, 255, 255), thickness, cv2.LINE_AA)
        
        # Bottom left: Copyright (small font)
        copyright_text = "(c) Prof. Shahab Anbarjafari - 3S Holding"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        thickness = 1
        
        padding = 10
        x = padding
        y = h - padding
        
        # Draw semi-transparent background
        (text_width, text_height), baseline = cv2.getTextSize(
            copyright_text, font, font_scale, thickness
        )
        overlay = frame.copy()
        cv2.rectangle(overlay,
                     (x - 5, y - text_height - 5),
                     (x + text_width + 5, y + baseline + 5),
                     (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        
        cv2.putText(frame, copyright_text, (x, y), font, font_scale,
                   (200, 200, 200), thickness, cv2.LINE_AA)
        
        return frame
    
    def run(self):
        """Main tracking loop."""
        print("\n" + "="*60)
        print("Gaze Tracking Service Started")
        print("="*60)
        print(f"Saving gaze data to: {self.json_filename}")
        print("Press 'q' to quit, 'c' to recalibrate")
        print("="*60 + "\n")
        
        frame_count = 0
        fps_start_time = time.time()
        fps = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Failed to capture frame")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = self.face_mesh.process(frame_rgb)
            
            screen_x, screen_y = None, None
            features = None
            
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                
                # Extract eye features
                features = self.get_eye_features(landmarks, frame.shape)
                
                if features is not None and self.is_calibrated:
                    # Estimate gaze point on screen
                    screen_x, screen_y = self.estimate_gaze_point(features)
                    
                    # Draw eye visualization
                    h, w = frame.shape[:2]
                    
                    # Draw eye landmarks
                    for idx in self.LEFT_EYE_INDICES + self.RIGHT_EYE_INDICES:
                        if idx < len(landmarks):
                            lm = landmarks[idx]
                            x, y = int(lm.x * w), int(lm.y * h)
                            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                    
                    # Draw iris
                    for idx in self.LEFT_IRIS_INDICES + self.RIGHT_IRIS_INDICES:
                        if idx < len(landmarks):
                            lm = landmarks[idx]
                            x, y = int(lm.x * w), int(lm.y * h)
                            cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)
            
            # Save gaze data
            if screen_x is not None and screen_y is not None:
                self.save_gaze_data(screen_x, screen_y, features)
            
            # Add text overlays
            frame = self.add_text_overlay(frame, screen_x, screen_y)
            
            # Calculate and display FPS
            frame_count += 1
            if frame_count >= 30:
                fps = 30 / (time.time() - fps_start_time)
                fps_start_time = time.time()
                frame_count = 0
            
            # Status indicator
            if self.is_calibrated:
                status_text = f"FPS: {fps:.1f} | CALIBRATED"
                status_color = (0, 255, 0)
            else:
                status_text = f"FPS: {fps:.1f} | NOT CALIBRATED (Press 'c')"
                status_color = (0, 0, 255)
            
            cv2.putText(frame, status_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
            
            # Display frame
            cv2.imshow("Gaze Tracking - Eye View", frame)
            
            # Check for quit or recalibrate
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC
                break
            elif key == ord('c'):  # Recalibrate
                print("\nRecalibrating...")
                self.run_calibration()
        
        # Cleanup
        self.cleanup()
    
    def cleanup(self):
        """Release resources."""
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        print(f"\nGaze tracking data saved to: {self.json_filename}")
        print(f"Total samples collected: {len(self.gaze_history)}")
        print("\nThank you for using Gaze Tracking Service!")


def main():
    """Main entry point."""
    print("="*60)
    print("Gaze Tracking Service with Calibration")
    print("(c) Prof. Shahab Anbarjafari - 3S Holding")
    print("="*60)
    
    service = GazeTrackingService()
    
    # Select monitor
    service.select_monitor()
    
    # Initialize camera
    service.initialize_camera()
    
    # Run calibration
    service.run_calibration()
    
    if not service.is_calibrated:
        print("\nError: Calibration failed. Cannot proceed without calibration.")
        print("Please ensure:")
        print("  - Your face is clearly visible to the camera")
        print("  - You have good lighting")
        print("  - You look directly at each calibration dot")
        service.cleanup()
        return
    
    # Run tracking
    try:
        service.run()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        service.cleanup()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        service.cleanup()


if __name__ == "__main__":
    main()
