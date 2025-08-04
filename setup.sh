#!/bin/bash

echo "🚀 Khởi động hệ thống chẩn đoán bệnh ngoài da..."

# Kiểm tra Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 chưa được cài đặt!"
    exit 1
fi

echo "✅ Python3 đã sẵn sàng"

# Tạo virtual environment nếu chưa có
if [ ! -d "venv" ]; then
    echo "📦 Tạo virtual environment..."
    python3 -m venv venv
fi

# Kích hoạt virtual environment
echo "🔧 Kích hoạt virtual environment..."
source venv/bin/activate

# Cài đặt dependencies
echo "📥 Cài đặt dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Tạo thư mục logs nếu chưa có
mkdir -p logs

echo "✅ Hoàn tất cài đặt!"
echo ""
echo "🌟 Hướng dẫn khởi chạy:"
echo "1. Terminal 1 - Khởi chạy FastAPI backend:"
echo "   uvicorn app:app --reload --host 0.0.0.0 --port 8000"
echo ""
echo "2. Terminal 2 - Khởi chạy Streamlit frontend:"
echo "   streamlit run streamlit_app.py --server.port 8501"
echo ""
echo "3. Truy cập ứng dụng tại:"
echo "   - Backend API: http://localhost:8000"
echo "   - Frontend UI: http://localhost:8501"
echo ""
echo "🎉 Chúc bạn sử dụng thành công!"
