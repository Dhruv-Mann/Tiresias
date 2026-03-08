#!/bin/bash
# ============================================================
# Tiresias - Virtual Environment Setup Script (macOS/Linux)
# ============================================================
# This script creates a Python 3.12 virtual environment,
# activates it, and installs all required dependencies.
# ============================================================

echo "[Tiresias] Creating virtual environment with Python 3.12..."
python3.12 -m venv venv

if [ ! -f "venv/bin/activate" ]; then
    echo "[ERROR] Failed to create virtual environment. Is Python 3.12 installed?"
    exit 1
fi

echo "[Tiresias] Activating virtual environment..."
source venv/bin/activate

echo "[Tiresias] Upgrading pip..."
python -m pip install --upgrade pip

echo "[Tiresias] Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "============================================================"
echo "[Tiresias] Setup complete!"
echo "To activate the environment later, run:"
echo "    source venv/bin/activate"
echo "============================================================"
