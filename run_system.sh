#!/bin/bash

# 🚀 Run Skin Cancer Classification System

VENV_PATH="/home/dekiru/Desktop/env"
PROJECT_PATH="/home/dekiru/Desktop/Onschool/profession_project/skin_cancer_classification"

echo "🏥 Starting Skin Cancer Classification System..."

# Kill existing processes
pkill -f "uvicorn.*app:app" 2>/dev/null || true
pkill -f "streamlit.*streamlit_app.py" 2>/dev/null || true

sleep 2

# Start FastAPI backend
echo "🚀 Starting FastAPI backend (port 8001)..."
cd "$PROJECT_PATH"
"$VENV_PATH/bin/python3.10" -m uvicorn app:app --port 8001 --reload &
BACKEND_PID=$!

# Wait for backend to start
echo "⏳ Waiting for backend to start..."
sleep 8

# Test backend
if curl -s http://localhost:8001/ > /dev/null; then
    echo "✅ Backend started successfully"
else
    echo "❌ Backend failed to start"
    kill $BACKEND_PID 2>/dev/null || true
    exit 1
fi

# Start Streamlit frontend  
echo "🎨 Starting Streamlit frontend (port 8502)..."
"$VENV_PATH/bin/python3.10" -m streamlit run streamlit_app.py --server.port 8502 --server.fileWatcherType none &
FRONTEND_PID=$!

echo ""
echo "🎉 System started successfully!"
echo ""
echo "📡 API Backend:  http://localhost:8001"
echo "🎨 Web Frontend: http://localhost:8502"
echo ""
echo "💡 To stop the system:"
echo "   pkill -f uvicorn"  
echo "   pkill -f streamlit"
echo ""

# Keep script running
wait
