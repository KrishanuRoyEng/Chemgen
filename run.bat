@echo off
REM Activate virtual environment
call .venv\Scripts\activate.bat

REM Start Backend in a new window
start "Backend Server" python backend\main.py

REM Start Frontend in this window
cd frontend
npm start
