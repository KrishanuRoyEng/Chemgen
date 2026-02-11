#!/bin/bash

# Define cleanup function to kill background processes
cleanup() {
    echo "Stopping backend..."
    kill $BACKEND_PID
    exit
}

# Trap SIGINT (Ctrl+C) to run cleanup
trap cleanup SIGINT

# 1. Start Virtual Environment
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python -m venv .venv
fi
echo "Activating virtual environment..."
source .venv/Scripts/activate

# 2. Install Python Dependencies
echo "Installing/Updating Python dependencies..."
pip install -r backend/requirements.txt

# 3. Start Backend
echo "Starting Backend..."
python backend/main.py &
BACKEND_PID=$!

# Wait a moment for backend to initialize
sleep 2

# 4. Install Frontend Dependencies & Start
echo "Starting Frontend..."
cd frontend
if [ ! -d "node_modules" ]; then
    echo "Installing Frontend dependencies..."
    npm install
fi
npm start
