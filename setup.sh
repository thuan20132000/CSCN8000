#!/bin/bash

# Statistics Practice Environment Setup Script
# This script automates the setup process

echo "=========================================="
echo "Statistics Practice Environment Setup"
echo "=========================================="
echo ""

# Check Python installation
echo "Checking Python installation..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    echo "✓ $PYTHON_VERSION found"
else
    echo "✗ Python 3 not found. Please install Python 3.8 or higher."
    exit 1
fi

# Create virtual environment
echo ""
echo "Creating virtual environment..."
if [ -d "venv" ]; then
    echo "✓ Virtual environment already exists"
else
    python3 -m venv venv
    echo "✓ Virtual environment created"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip --quiet
echo "✓ pip upgraded"

# Install requirements
echo ""
echo "Installing required packages..."
echo "This may take a few minutes..."
pip install -r requirements.txt --quiet
echo "✓ All packages installed"

# Install Jupyter kernel
echo ""
echo "Installing Jupyter kernel..."
python -m ipykernel install --user --name=stats-practice --display-name="Python (Stats Practice)"
echo "✓ Jupyter kernel installed"

# Summary
echo ""
echo "=========================================="
echo "Setup Complete! 🎉"
echo "=========================================="
echo ""
echo "To get started:"
echo "  1. Activate the environment:  source venv/bin/activate"
echo "  2. Launch Jupyter:            jupyter notebook"
echo "  3. Open getting_started.ipynb to verify setup"
echo ""
echo "For more information, see SETUP_GUIDE.md"
echo ""
