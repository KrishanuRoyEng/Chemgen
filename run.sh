#!/bin/bash

# Define cleanup function to kill background processes
cleanup() {
    echo "Stopping backend..."
    kill $BACKEND_PID
    exit
}

# Trap SIGINT (Ctrl+C) to run cleanup
trap cleanup SIGINT

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/Scripts/activate

# Start Backend
echo "Starting Backend..."
python backend/main.py &
BACKEND_PID=$!

# Wait a moment for backend to initialize
sleep 2

# Start Frontend
echo "Starting Frontend..."
cd frontend
npm start
