#!/bin/bash

# Eye Tracking Service - Run Script
# (c) Prof. Shahab Anbarjafari - 3S Holding

# Check if virtual environment exists
if [ ! -d "myenv" ]; then
    echo "Error: Virtual environment not found!"
    echo "Please run ./install.sh first"
    exit 1
fi

# Activate virtual environment
source myenv/bin/activate

# Run the eye tracking service
python eye_tracking_service.py

# Deactivate when done
deactivate

