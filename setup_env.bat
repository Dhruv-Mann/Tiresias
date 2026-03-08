@echo off
REM ============================================================
REM Tiresias - Virtual Environment Setup Script (Windows)
REM ============================================================
REM This script creates a Python 3.12 virtual environment,
REM activates it, and installs all required dependencies.
REM ============================================================

echo [Tiresias] Creating virtual environment with Python 3.12...
py -3.12 -m venv venv

if not exist "venv\Scripts\activate.bat" (
    echo [ERROR] Failed to create virtual environment. Is Python 3.12 installed?
    pause
    exit /b 1
)

echo [Tiresias] Activating virtual environment...
call venv\Scripts\activate.bat

echo [Tiresias] Upgrading pip...
python -m pip install --upgrade pip

echo [Tiresias] Installing dependencies...
pip install -r requirements.txt

echo.
echo ============================================================
echo [Tiresias] Setup complete!
echo To activate the environment later, run:
echo     venv\Scripts\activate
echo ============================================================
pause
