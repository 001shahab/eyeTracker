#!/bin/bash

# Eye Tracking Service - Installation Script
# (c) Prof. Shahab Anbarjafari - 3S Holding

echo "=========================================="
echo "Eye Tracking Service - Installation"
echo "(c) Prof. Shahab Anbarjafari - 3S Holding"
echo "=========================================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed!"
    echo "Please install Python 3.8 or higher from https://www.python.org/"
    exit 1
fi

echo "✓ Python found: $(python3 --version)"
echo ""

# Check if virtual environment exists
if [ ! -d "myenv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv myenv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source myenv/bin/activate
echo "✓ Virtual environment activated"
echo ""

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip
echo ""

# Install requirements
echo "Installing dependencies..."
echo "This may take a few minutes..."
pip install -r requirements.txt
echo ""

echo "=========================================="
echo "✓ Installation Complete!"
echo "=========================================="
echo ""
echo "To start the eye tracking service:"
echo "  1. source myenv/bin/activate"
echo "  2. python eye_tracking_service.py"
echo ""
echo "Or simply run: ./run.sh"
echo ""

