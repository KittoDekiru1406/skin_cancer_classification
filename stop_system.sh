#!/bin/bash

echo "🛑 Stopping Skin Cancer Classification System..."

# Stop processes
pkill -f "uvicorn.*app:app" 2>/dev/null && echo "✅ Backend stopped" || echo "ℹ️ Backend not running"
pkill -f "streamlit.*streamlit_app.py" 2>/dev/null && echo "✅ Frontend stopped" || echo "ℹ️ Frontend not running"

echo "🏁 System stopped"
