@echo off
setlocal

:: Set filenames and paths
set PYTHON_EXE=python-3.11.0-amd64.exe
set PYTHON_DIR=%~dp0python
set VENV_DIR=%~dp0venv
set BEST_PT=best.pt

echo =========================================
echo Checking Python installation...
echo =========================================

:: Check if python folder exists
if not exist "%PYTHON_DIR%" (
    echo Python not found, downloading...

    :: Download Python installer from Google Drive
    pip install gdown >nul 2>&1
    gdown --id 1df6HarCE5XC3GeccFjpRLkJLXgEdilG7 -O %PYTHON_EXE%

    echo Installing Python locally...
    %PYTHON_EXE% /quiet InstallAllUsers=0 PrependPath=0 TargetDir=%PYTHON_DIR%
    del %PYTHON_EXE%
) else (
    echo Python already installed locally.
)

:: Set local Python executable
set PY_EXE=%PYTHON_DIR%\python.exe

echo =========================================
echo Creating virtual environment...
echo =========================================

if not exist "%VENV_DIR%" (
    "%PY_EXE%" -m venv venv
)

echo Activating virtual environment...
call "%VENV_DIR%\Scripts\activate.bat"

echo =========================================
echo Installing dependencies...
echo =========================================

pip install --upgrade pip setuptools wheel build >nul
pip install -r requirements.txt

echo =========================================
echo Downloading YOLO model (best.pt)...
echo =========================================

if not exist "%BEST_PT%" (
    gdown --id 1CYYfDl1yZ7v6UYSligIGfRuxx0KllBW2 -O best.pt
) else (
    echo best.pt already exists.
)

echo =========================================
echo Running main.py...
echo =========================================

python main.py

pause
