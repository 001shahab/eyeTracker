#!/bin/bash

# Test script to demonstrate analysis tools
# (c) Prof. Shahab Anbarjafari - 3S Holding

echo "=========================================="
echo "Gaze Tracking Analysis Demo"
echo "=========================================="
echo ""

# Find the most recent JSON file in results/
LATEST_JSON=$(ls -t results/gaze_tracking_*.json 2>/dev/null | head -n 1)

if [ -z "$LATEST_JSON" ]; then
    echo "No gaze tracking data found in results/ folder."
    echo ""
    echo "Please run the eye tracking service first:"
    echo "  ./run.sh"
    echo ""
    exit 1
fi

echo "Found tracking data: $LATEST_JSON"
echo ""

# Activate virtual environment
source myenv/bin/activate

# Run analysis
echo "=========================================="
echo "1. Statistical Analysis"
echo "=========================================="
echo ""
python analyze_gaze_data.py "$LATEST_JSON"
echo ""
echo ""

# Run visualization
echo "=========================================="
echo "2. Creating Visualizations"
echo "=========================================="
echo ""
python visualize_gaze_heatmap.py "$LATEST_JSON"
echo ""

echo "=========================================="
echo "Analysis Complete!"
echo "=========================================="
echo ""
echo "Check the results/ folder for:"
echo "  - $LATEST_JSON (raw data)"
echo "  - results/gaze_heatmap.png (attention heatmap)"
echo "  - results/gaze_trajectory.png (gaze path)"
echo ""

