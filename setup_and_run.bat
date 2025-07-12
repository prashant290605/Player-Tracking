@echo off

echo Installing Python...
:: Assume python311 is extracted and in current folder

echo Creating virtual environment...
python311\python.exe -m venv venv

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Installing pip tools...
python -m ensurepip --upgrade
python -m pip install --upgrade pip setuptools wheel build

echo Installing dependencies...
pip install -r requirements.txt

echo Running project...
python main.py

pause
