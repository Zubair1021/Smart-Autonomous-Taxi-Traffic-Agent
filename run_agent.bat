
@echo off
REM ============================================================
REM Smart Autonomous Taxi & Traffic System - Auto Setup & Run
REM ============================================================

echo ============================================================
echo Smart Autonomous Taxi & Traffic System
echo Professional Dashboard - Auto Setup
echo ============================================================
echo.

REM [1/5] Verifying Python installation...
echo [1/5] Verifying Python version...
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH!
    echo Please install Python 3.11 or higher from https://www.python.org/
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo Python version: %PYTHON_VERSION%
echo [OK] Python found
echo.

REM [2/5] Checking virtual environment...
echo [2/5] Checking virtual environment...
if exist "venv\Scripts\activate.bat" (
    echo [INFO] Virtual environment found
    goto :activate_venv
) else (
    echo [INFO] Virtual environment not found
    echo Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment!
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created
)

:activate_venv
echo Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment!
    pause
    exit /b 1
)
echo [OK] Virtual environment activated
echo.

REM [3/5] Checking requirements file...
echo [3/5] Checking requirements file...
if not exist "requirements.txt" (
    echo [ERROR] requirements.txt not found!
    pause
    exit /b 1
)
echo [OK] requirements.txt found
echo.

REM [4/5] Checking and installing dependencies...
echo [4/5] Checking and installing dependencies...
echo This may take a few moments...
echo.

echo Upgrading pip...
python -m pip install --upgrade pip --quiet

echo Installing required packages...
pip install -r requirements.txt
if errorlevel 1 (
    echo [WARNING] Standard installation failed, trying with compatible versions...
    echo This may be due to Python 3.13 compatibility issues.
    echo Installing updated compatible packages...
    pip install --upgrade pip setuptools wheel
    pip install -r requirements.txt --no-cache-dir
    if errorlevel 1 (
        echo [ERROR] Failed to install requirements!
        echo Troubleshooting:
        echo 1. Python 3.13 may have compatibility issues with some packages
        echo 2. Consider using Python 3.11 or 3.12 for best compatibility
        echo 3. Or install Visual Studio Build Tools for compiling packages
        pause
        exit /b 1
    )
)

REM Verify critical packages
echo.
echo Verifying critical packages...
python -c "import dash" 2>nul
if errorlevel 1 (
    echo [ERROR] dash not installed correctly!
    pause
    exit /b 1
)

python -c "import dash_bootstrap_components" 2>nul
if errorlevel 1 (
    echo [ERROR] dash_bootstrap_components not installed correctly!
    pause
    exit /b 1
)

python -c "import plotly" 2>nul
if errorlevel 1 (
    echo [ERROR] plotly not installed correctly!
    pause
    exit /b 1
)

echo [OK] All critical packages verified
echo.

REM [5/5] Starting dashboard...
echo [5/5] Starting dashboard...
echo.
echo ============================================================
echo Starting Smart Autonomous Taxi and Traffic System...
echo ============================================================
echo The simulation will open in your browser at: http://127.0.0.1:8050
echo Press Ctrl+C to stop the simulation
echo ============================================================
echo.

python dashboard_main.py

echo.
echo ============================================================
echo Simulation ended.
echo ============================================================
pause

