@echo off

REM 1. Start Virtual Environment
if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
)

echo Activating virtual environment...
call .venv\Scripts\activate.bat

REM 2. Install Python Dependencies
echo Installing/Updating Python dependencies...
pip install -r backend\requirements.txt

REM 3. Start Backend in a new window
echo Starting Backend...
start "Backend Server" python backend\main.py

REM 4. Install Frontend Dependencies & Start
cd frontend
if not exist "node_modules" (
    echo Installing Frontend dependencies...
    call npm install
)

echo Starting Frontend...
npm start
