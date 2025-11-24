# Gaze Tracking Service

**Professional Gaze Tracking System with Calibration and Multi-Monitor Support**

*(c) Prof. Shahab Anbarjafari - 3S Holding*

---

## 🚀 Quick Start (5 Minutes)

### Installation

```bash
# Navigate to project directory
cd /Users/shahab/Desktop/MySandBox/eyeTracker

# Run installation script
./install.sh
```

### Run the Service

```bash
# Option 1: Using the run script (Recommended)
./run.sh

# Option 2: Manual activation
source myenv/bin/activate
python eye_tracking_service.py
```

### What Happens

1. **Monitor Selection**: Choose which monitor to track (if you have multiple)
2. **Calibration**: Look at red dots that appear on screen until they turn green
3. **Tracking**: The system now knows where you're looking on screen!
4. **Data**: All gaze data is saved to `results/gaze_tracking_YYYYMMDD_HHMMSS.json`

### Controls

- **Q** or **ESC**: Quit tracking
- **C**: Recalibrate

### Quick Tips

- Sit 40-60 cm from camera
- Face camera directly with good lighting
- Keep head relatively still during calibration
- Clean camera lens for best accuracy

---

## 🎯 Features

- **Calibrated Gaze Tracking**: Knows WHERE on screen you're looking (not just eye movement)
- **9-Point Calibration**: Personal calibration for accurate gaze-to-screen mapping
- **Multi-Monitor Support**: Automatically detects all monitors and lets you choose
- **Real-Time Tracking**: 30+ FPS performance with live coordinate display
- **Dynamic JSON Logging**: Continuously updated gaze data in `results/` folder
- **Visual Feedback**: 
  - Eye landmarks and iris position visualization
  - Gaze coordinates displayed (bottom right, medium font)
  - Copyright notice (bottom left, small font)
- **RBF Interpolation**: Smooth, accurate gaze estimation using Radial Basis Functions
- **Recalibration**: Press 'C' anytime to recalibrate

---

## 📋 Requirements

- Python 3.8 or higher
- Webcam (built-in or external)
- macOS, Windows, or Linux

---

## 💻 Detailed Usage

### Step-by-Step Process

#### 1. Monitor Selection

If you have multiple monitors:

```
Multiple monitors detected. Please select one:
[0] Monitor: Display 1
    Resolution: 1920x1080
    Position: (0, 0)
    [PRIMARY]

[1] Monitor: Display 2
    Resolution: 2560x1440
    Position: (1920, 0)

Select monitor [0-1]: 
```

Type the number and press Enter.

#### 2. Calibration Phase

**This is the KEY difference from basic eye tracking!**

You'll see:
- Red dots appear at 9 positions on your screen (3x3 grid)
- Look directly at each dot
- Dot turns green with progress ring as samples are collected
- Repeat for all 9 points

**Important**: 
- Keep your head still during calibration
- Look ONLY with your eyes, not by moving your head
- The system learns YOUR specific gaze patterns

#### 3. Tracking Phase

After calibration, the system will:
- Show your webcam feed with eye visualization
- Display real-time gaze coordinates where you're looking
- Save all data to JSON file in `results/` folder
- Update JSON dynamically (can view while tracking)

**What You'll See**:
- Green dots: Eye landmarks
- Blue dots: Iris positions
- **Bottom Right (medium font)**: "Gaze: (X, Y)" - where you're looking on screen
- **Bottom Left (small font)**: "(c) Prof. Shahab Anbarjafari - 3S Holding"
- **Top Left**: FPS and calibration status

---

## 📊 Output Data Format

All data is saved to `results/gaze_tracking_YYYYMMDD_HHMMSS.json`:

```json
{
  "session_start": "2023-11-24T14:30:22.123456",
  "total_samples": 1500,
  "calibration_points": [
    [320, 180],
    [960, 180],
    [1600, 180],
    [320, 540],
    [960, 540],
    [1600, 540],
    [320, 900],
    [960, 900],
    [1600, 900]
  ],
  "gaze_data": [
    {
      "timestamp": "2023-11-24T14:30:22.123456",
      "screen_x": 1250,
      "screen_y": 680,
      "eye_features": [0.02, -0.01, 0.021, -0.009, 0.019, -0.011, 0.48, 0.52, 0.15],
      "monitor": {
        "width": 1920,
        "height": 1080,
        "x": 0,
        "y": 0
      }
    }
  ]
}
```

### Data Fields

- **timestamp**: ISO format timestamp
- **screen_x, screen_y**: ACTUAL screen coordinates where you're looking
- **eye_features**: 9-dimensional feature vector used for gaze estimation
  - Average iris offset (x, y)
  - Left eye iris offset (x, y)
  - Right eye iris offset (x, y)
  - Head pose indicators (nose position, tilt)
- **calibration_points**: Screen positions used during calibration
- **monitor**: Display information

---

## 📈 Data Analysis & Visualization

### Quick Analysis (Easiest Way)

```bash
# Analyzes the most recent tracking session automatically
./test_analysis.sh
```

This will:
1. Find the latest JSON file in results/
2. Run statistical analysis
3. Generate heatmap and trajectory visualizations

**Example Output:**
```
GAZE TRACKING DATA ANALYSIS
📊 Data Format: Calibrated Gaze Tracking (with personal calibration)

SESSION INFORMATION:
  Duration:      16.0 seconds
  Total Samples: 472
  Sampling Rate: 29.5 Hz

CALIBRATION INFORMATION:
  Calibration Points: 9
  Calibration Grid: (9-point 3x3 layout shown)

SCREEN REGION ANALYSIS:
  Bottom-Right   :   283 samples ( 60.0%) ███████████████████
  Bottom-Center  :   118 samples ( 25.0%) ████████████
  Bottom-Left    :    69 samples ( 14.6%) ███████
  (Shows where you looked most on screen)
```

### Analyze Specific Session

```bash
python analyze_gaze_data.py results/gaze_tracking_20231124_143022.json
```

Shows:
- Data format detection (calibrated vs basic)
- Session statistics (duration, sampling rate)
- Calibration information (9-point grid)
- Screen coordinate statistics (min, max, mean, std dev)
- Screen region analysis (3x3 grid showing where you looked most)
- Eye feature analysis (9D feature vector statistics)
- Primary fixation points

### Visualize as Heatmap & Trajectory

```bash
python visualize_gaze_heatmap.py results/gaze_tracking_20231124_143022.json
```

Creates:
- `results/gaze_heatmap.png`: Heatmap showing attention density
  - Color-coded: Red/yellow = high attention, blue = low attention
  - Includes calibration point markers (cyan dots with numbers)
  - Shows where you looked most frequently
- `results/gaze_trajectory.png`: Path of eye movement over time
  - Color gradient: Blue (start) → Red (end)
  - Shows the path your gaze took during the session
  - Faint gray markers show calibration points

---

## 🎯 How It Works (Technical)

### The Problem with Basic Eye Tracking

Basic eye tracking only measures iris position relative to eye center. This tells you which direction eyes are pointing, but NOT where on screen the person is looking.

**Example**: You can be looking at top-right of screen with:
- Eyes centered + head tilted
- Eyes looking right + head straight
- Various combinations!

### The Solution: Calibrated Gaze Tracking

Our system:

1. **Calibration Phase**: You look at known screen positions
2. **Feature Extraction**: System records eye features at those positions
3. **Model Building**: RBF interpolation creates mapping from features → screen coordinates
4. **Tracking Phase**: Apply model to estimate gaze point on screen

### Architecture

```
Webcam → MediaPipe Face Mesh (478 landmarks + iris)
       ↓
Extract Eye Features (9D vector)
  - Iris offsets (both eyes)
  - Head pose indicators
       ↓
[CALIBRATION PHASE]
  User looks at 9 known points
  Build RBF interpolation model
       ↓
[TRACKING PHASE]
  Apply model to eye features
  → Screen coordinates (X, Y)
       ↓
Display + Save to results/
```

### Key Components

1. **MediaPipe Face Mesh**: 478 facial landmarks + iris tracking
2. **Feature Extraction**: 9D vector capturing gaze direction and head pose
3. **RBF Interpolation**: Smooth, non-linear mapping from eye features to screen coordinates
4. **Calibration**: Personal model for each user and seating position

### Accuracy

Expected accuracy with proper calibration:
- **±30-80 pixels** on 1920x1080 display under good conditions
- Significantly better than uncalibrated approaches
- Accuracy depends on:
  - Quality of calibration (looking directly at dots)
  - Head stability during tracking
  - Lighting conditions
  - Camera quality

---

## 🔧 Technical Details

### Landmark Indices Used

- **Left Eye**: 33, 133, 160, 159, 158, 157, 173, 246
- **Right Eye**: 362, 263, 387, 386, 385, 384, 398, 466
- **Left Iris**: 468, 469, 470, 471, 472
- **Right Iris**: 473, 474, 475, 476, 477
- **Head Pose**: 1 (nose tip), 152 (chin)

### Feature Vector (9D)

```python
features = [
    avg_iris_offset_x,      # Average horizontal iris offset
    avg_iris_offset_y,      # Average vertical iris offset
    left_iris_offset_x,     # Left eye horizontal
    left_iris_offset_y,     # Left eye vertical
    right_iris_offset_x,    # Right eye horizontal
    right_iris_offset_y,    # Right eye vertical
    nose_x,                 # Head pose horizontal
    nose_y,                 # Head pose vertical
    chin_nose_distance      # Head tilt indicator
]
```

### RBF Interpolation

Uses `scipy.interpolate.Rbf` with:
- Function: 'multiquadric' (smooth, continuous)
- Smoothing: 0.1 (prevents overfitting)
- Separate models for X and Y coordinates

---

## 🐛 Troubleshooting

### Calibration Issues

**Calibration fails or times out:**
- Ensure face is clearly visible
- Improve lighting
- Face camera directly
- Look at dots with your EYES, not by moving head
- Keep head still during calibration

**Poor accuracy after calibration:**
- Recalibrate (press 'C')
- Improve lighting conditions
- Reduce head movement
- Sit at same position as during calibration
- Clean camera lens

### Camera Issues

**Camera won't open:**
- Close apps using camera (Zoom, Skype, etc.)
- Check camera permissions in System Settings
- Try different camera index in code

**No face detected:**
- Improve lighting
- Move closer (40-60cm optimal)
- Face camera directly
- Remove obstructions

### Performance Issues

**Low FPS:**
- Close other applications
- Use better hardware
- Reduce camera resolution in code

---

## 📁 Project Structure

```
eyeTracker/
├── eye_tracking_service.py      # Main gaze tracking service with calibration
├── analyze_gaze_data.py          # Statistical analysis tool
├── visualize_gaze_heatmap.py    # Visualization tool (heatmap + trajectory)
├── test_analysis.sh              # Quick analysis of latest session
├── requirements.txt              # Python dependencies
├── install.sh                    # Installation script
├── run.sh                        # Quick run script
├── README.md                     # This file
├── .gitignore                    # Git configuration
├── myenv/                        # Virtual environment
└── results/                      # Output folder
    ├── .gitkeep                  # Keeps folder in git
    ├── gaze_tracking_*.json      # Gaze data (timestamped)
    ├── gaze_heatmap.png          # Heatmap visualization
    └── gaze_trajectory.png       # Trajectory visualization
```

---

## 💡 Use Cases

- **UI/UX Research**: Understand where users look on interfaces
- **Accessibility**: Hands-free computer control
- **Medical Research**: Neurological assessments, attention studies
- **Gaming**: Gaze-based interactions and controls
- **Marketing**: Advertisement effectiveness, attention tracking
- **Education**: Student attention and engagement monitoring
- **Usability Testing**: Heat maps of visual attention
- **Human-Computer Interaction**: Novel interaction paradigms

---

## 🎓 For Researchers

### Calibration Best Practices

1. **Environment Consistency**: Calibrate in the same conditions as tracking
2. **Multiple Calibrations**: Average results from 2-3 calibration runs
3. **Periodic Recalibration**: Recalibrate if head position changes
4. **Participant Instructions**: Clear instructions on looking AT dots, not around them

### Data Collection Tips

1. **Pilot Testing**: Test setup with several participants first
2. **Calibration Validation**: Show known targets after calibration to verify accuracy
3. **Logging**: JSON format allows easy import to analysis tools
4. **Synchronization**: Timestamp format allows sync with other data sources

### Extending the System

The code is modular and can be extended:
- Add more calibration points (modify `cal_points` list)
- Change interpolation method (try 'thin_plate', 'gaussian', etc.)
- Add temporal smoothing (Kalman filter, moving average)
- Implement blink detection
- Add head pose compensation

---

## 📝 License & Citation

(c) Prof. Shahab Anbarjafari - 3S Holding

For academic use, please cite appropriately.

---

## 🚧 Future Enhancements

Potential improvements:

1. **Enhanced Calibration**: 
   - 13-point or 16-point calibration for corners
   - Validation phase to verify accuracy
   - Adaptive calibration based on initial accuracy

2. **Machine Learning**: 
   - Deep learning model for gaze estimation
   - Transfer learning from other users
   - Online learning and adaptation

3. **Robustness**:
   - Kalman filtering for smooth trajectories
   - Outlier detection and removal
   - Blink detection and handling
   - Head pose compensation

4. **Features**:
   - Multi-person tracking
   - Fixation detection algorithms
   - Saccade analysis
   - AOI (Area of Interest) tools

5. **Integration**:
   - Real-time streaming to other applications
   - Plugin architecture
   - Web-based interface

---

## 📞 Support

For questions, issues, or contributions:
- Prof. Shahab Anbarjafari
- 3S Holding

---

**Now you're ready to track gazes accurately! 👁️✨**

Remember: **Calibration is key!** The system learns YOUR specific gaze patterns to provide accurate screen coordinates.
