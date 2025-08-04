"""
Streamlit Frontend cho hệ thống chẩn đoán bệnh ngoài da
"""

import streamlit as st
import requests
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import io
import json
import time
from typing import Dict, Any

# Cấu hình trang
st.set_page_config(
    page_title="🏥 Skin Cancer Classification System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# URL của FastAPI backend
API_BASE_URL = "http://localhost:8001"

# Custom CSS để làm đẹp giao diện
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .disease-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid;
        margin: 0.5rem 0;
    }
    
    .benign { border-left-color: #28a745; }
    .malignant { border-left-color: #dc3545; }
    .precancer { border-left-color: #ffc107; }
    .cancer { border-left-color: #fd7e14; }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    .stProgress .st-bo {
        background-color: #e9ecef;
    }
    
    .upload-box {
        border: 2px dashed #cccccc;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

def display_header():
    """Hiển thị header của ứng dụng"""
    st.markdown("""
    <div class="main-header">
        <h1>🏥 Hệ Thống Chẩn Đoán Bệnh Ngoài Da</h1>
        <h3>Sử dụng Mạng Nơ-ron Tích Chập (CNN)</h3>
        <p>Phân loại 7 loại bệnh da phổ biến với độ chính xác cao</p>
    </div>
    """, unsafe_allow_html=True)

def check_api_health():
    """Kiểm tra trạng thái API"""
    try:
        response = requests.get(f"{API_BASE_URL}/", timeout=5)
        if response.status_code == 200:
            return True, response.json()
        return False, None
    except:
        return False, None

def get_available_models():
    """Lấy danh sách models có sẵn"""
    try:
        response = requests.get(f"{API_BASE_URL}/models", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

def get_disease_info():
    """Lấy thông tin về các loại bệnh"""
    try:
        response = requests.get(f"{API_BASE_URL}/disease-info", timeout=5)
        if response.status_code == 200:
            return response.json()["disease_info"]
        return None
    except:
        return None

def predict_image(image_file, model_name=None):
    """Gửi ảnh để dự đoán"""
    try:
        files = {"file": image_file}
        
        if model_name:
            url = f"{API_BASE_URL}/predict/{model_name}"
        else:
            url = f"{API_BASE_URL}/predict/all"
        
        response = requests.post(url, files=files, timeout=30)
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Lỗi API: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Lỗi kết nối: {str(e)}")
        return None

def display_disease_info(disease_info):
    """Hiển thị thông tin về các loại bệnh"""
    st.subheader("📋 Thông Tin Các Loại Bệnh")
    
    # Tạo các columns để hiển thị
    for disease_code, info in disease_info.items():
        risk_class = "benign"
        if info["type"] == "Ác tính":
            risk_class = "malignant"
        elif info["type"] == "Tiền ung thư":
            risk_class = "precancer"
        elif info["type"] == "Ung thư":
            risk_class = "cancer"
        
        st.markdown(f"""
        <div class="disease-card {risk_class}">
            <h4>{disease_code.upper()} - {info['name']}</h4>
            <p><strong>Loại:</strong> {info['type']} | <strong>Nguy cơ:</strong> {info['risk']}</p>
            <p>{info['description']}</p>
        </div>
        """, unsafe_allow_html=True)

def display_prediction_results(results, image):
    """Hiển thị kết quả dự đoán"""
    
    if "individual_results" in results:
        # Kết quả từ tất cả models
        st.subheader("🔍 Kết Quả Dự Đoán Từ Tất Cả Models")
        
        # Hiển thị ảnh đã upload
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(image, caption="Ảnh đã tải lên", width=300)
        
        # Consensus prediction
        if results.get("consensus_prediction"):
            consensus = results["consensus_prediction"]
            consensus_info = results.get("consensus_info", {})
            
            st.success(f"🎯 **Kết Quả Chung:** {consensus.upper()} - {consensus_info.get('name', '')}")
            
            if consensus_info:
                risk_color = consensus_info.get('color', '#000000')
                st.markdown(f"""
                <div style="background-color: {risk_color}; color: white; padding: 1rem; border-radius: 8px; margin: 1rem 0;">
                    <h4>{consensus_info['name']}</h4>
                    <p><strong>Loại:</strong> {consensus_info['type']}</p>
                    <p><strong>Mức độ nguy cơ:</strong> {consensus_info['risk']}</p>
                    <p>{consensus_info['description']}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Kết quả chi tiết từng model
        st.subheader("📊 Kết Quả Chi Tiết Từng Model")
        
        # Tạo bảng so sánh
        comparison_data = []
        for model_name, result in results["individual_results"].items():
            if "predicted_class" in result:
                comparison_data.append({
                    "Model": model_name.replace('_', ' ').title(),
                    "Prediction": result["predicted_class"].upper(),
                    "Confidence": f"{result['confidence']:.2%}",
                    "Disease": result["disease_info"]["name"]
                })
        
        if comparison_data:
            df_comparison = pd.DataFrame(comparison_data)
            st.dataframe(df_comparison, use_container_width=True)
        
        # Biểu đồ so sánh confidence
        fig_comparison = go.Figure()
        
        for model_name, result in results["individual_results"].items():
            if "predicted_class" in result:
                fig_comparison.add_trace(go.Bar(
                    name=model_name.replace('_', ' ').title(),
                    x=[result["predicted_class"].upper()],
                    y=[result["confidence"]],
                    text=[f"{result['confidence']:.2%}"],
                    textposition='auto'
                ))
        
        fig_comparison.update_layout(
            title="So Sánh Độ Tin Cậy Giữa Các Models",
            xaxis_title="Prediction",
            yaxis_title="Confidence",
            barmode='group'
        )
        st.plotly_chart(fig_comparison, use_container_width=True)
        
        # Hiển thị chi tiết từng model
        st.subheader("🔬 Chi Tiết Từng Model")
        
        tabs = st.tabs([model.replace('_', ' ').title() for model in results["individual_results"].keys()])
        
        for i, (model_name, result) in enumerate(results["individual_results"].items()):
            with tabs[i]:
                if "predicted_class" in result:
                    display_single_prediction(result, model_name)
                else:
                    st.error(f"Lỗi với model {model_name}: {result.get('error', 'Unknown error')}")
    
    else:
        # Kết quả từ một model
        st.subheader("🔍 Kết Quả Dự Đoán")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(image, caption="Ảnh đã tải lên", width=300)
        
        with col2:
            display_single_prediction(results["result"], results["model_used"])

def display_single_prediction(result, model_name):
    """Hiển thị kết quả dự đoán từ một model"""
    
    predicted_class = result["predicted_class"]
    confidence = result["confidence"]
    disease_info = result["disease_info"]
    all_predictions = result["all_predictions"]
    
    # Hiển thị kết quả chính
    st.metric(
        label="Dự đoán",
        value=f"{predicted_class.upper()} - {disease_info['name']}",
        delta=f"Confidence: {confidence:.2%}"
    )
    
    # Thông tin bệnh
    risk_color = disease_info.get('color', '#000000')
    st.markdown(f"""
    <div style="background-color: {risk_color}; color: white; padding: 1rem; border-radius: 8px; margin: 1rem 0;">
        <h4>{disease_info['name']}</h4>
        <p><strong>Loại:</strong> {disease_info['type']}</p>
        <p><strong>Mức độ nguy cơ:</strong> {disease_info['risk']}</p>
        <p>{disease_info['description']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Biểu đồ xác suất tất cả classes
    st.subheader(f"📈 Xác Suất Tất Cả Loại Bệnh ({model_name.replace('_', ' ').title()})")
    
    # Sắp xếp theo xác suất
    sorted_predictions = sorted(all_predictions.items(), key=lambda x: x[1], reverse=True)
    
    classes = [item[0].upper() for item in sorted_predictions]
    probabilities = [item[1] for item in sorted_predictions]
    
    fig = px.bar(
        x=probabilities,
        y=classes,
        orientation='h',
        title=f"Xác Suất Dự Đoán - {model_name.replace('_', ' ').title()}",
        labels={'x': 'Xác suất', 'y': 'Loại bệnh'},
        color=probabilities,
        color_continuous_scale='viridis'
    )
    
    fig.update_traces(text=[f"{p:.2%}" for p in probabilities], textposition='inside')
    fig.update_layout(height=400)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Bảng chi tiết
    df_detailed = pd.DataFrame(sorted_predictions, columns=['Disease Code', 'Probability'])
    df_detailed['Probability'] = df_detailed['Probability'].apply(lambda x: f"{x:.4f}")
    df_detailed['Percentage'] = [f"{p:.2%}" for p in [item[1] for item in sorted_predictions]]
    
    st.dataframe(df_detailed, use_container_width=True)

def main():
    """Hàm chính của ứng dụng"""
    
    # Header
    display_header()
    
    # Kiểm tra kết nối API
    api_healthy, api_info = check_api_health()
    
    if not api_healthy:
        st.error("❌ Không thể kết nối với backend API. Vui lòng khởi động server trước!")
        st.info("Chạy lệnh: `./run_system.sh` hoặc `uvicorn app:app --host 0.0.0.0 --port 8001` để khởi động server")
        return
    
    st.success("✅ Kết nối API thành công!")
    
    # Sidebar
    with st.sidebar:
        st.header("🎛️ Cấu Hình")
        
        # Lấy thông tin models
        models_info = get_available_models()
        if models_info:
            available_models = list(models_info["available_models"].keys())
            st.success(f"🤖 {models_info['total_models']} models đã sẵn sàng")
            
            # Models này là real models, không phải mock
            
            # Tùy chọn model  
            model_display_names = []
            for model in available_models:
                if model == 'mobilenet':
                    model_display_names.append('MobileNet')
                elif model == 'resnet50':
                    model_display_names.append('ResNet50')
                elif model == 'self_build':
                    model_display_names.append('Self Build')
                else:
                    model_display_names.append(model.replace('_', ' ').title())
            
            model_choice = st.selectbox(
                "Chọn Model:",
                ["Tất cả models"] + model_display_names
            )
            
            if model_choice != "Tất cả models":
                # Chuyển đổi tên model về format gốc
                model_mapping = {
                    'MobileNet': 'mobilenet',
                    'ResNet50': 'resnet50', 
                    'Self Build': 'self_build'
                }
                selected_model = model_mapping.get(model_choice, model_choice.lower().replace(' ', '_'))
            else:
                selected_model = None
        else:
            st.error("❌ Không thể tải thông tin models")
            return
        
        st.divider()
        
        # Thông tin ứng dụng
        st.header("ℹ️ Thông Tin")
        st.markdown("""
        **Hệ thống chẩn đoán bệnh ngoài da**
        
        - 🎯 **Models**: MobileNet, ResNet50, Self-build CNN
        - 🏷️ **Classes**: 7 loại bệnh da
        - 📊 **Accuracy**: Cao độ chính xác
        - ⚡ **Speed**: Dự đoán nhanh chóng
        """)
        
        st.divider()
        
        # Warning
        st.warning("""
        ⚠️ **Lưu ý quan trọng:**
        
        Hệ thống này chỉ phục vụ mục đích nghiên cứu và học tập. 
        
        **KHÔNG thay thế** việc chẩn đoán y tế chuyên nghiệp!
        """)
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["🔍 Chẩn Đoán", "📋 Thông Tin Bệnh", "📊 Demo Dataset"])
    
    with tab1:
        st.header("📤 Tải Lên Ảnh Để Chẩn Đoán")
        
        uploaded_file = st.file_uploader(
            "Chọn ảnh da để chẩn đoán",
            type=['png', 'jpg', 'jpeg'],
            help="Hỗ trợ định dạng: PNG, JPG, JPEG"
        )
        
        if uploaded_file is not None:
            try:
                # Validate image
                image = Image.open(uploaded_file)
                
                # Check image properties
                if image.size[0] < 50 or image.size[1] < 50:
                    st.warning("⚠️ Ảnh quá nhỏ. Khuyến nghị ảnh có kích thước ít nhất 224x224 pixels.")
                
                # Display image info
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.image(image, caption=f"Ảnh đã tải lên | Kích thước: {image.size[0]}x{image.size[1]}", width=300)
                
                # Image info metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Width", f"{image.size[0]}px")
                with col2:
                    st.metric("Height", f"{image.size[1]}px")
                with col3:
                    st.metric("Mode", image.mode)
                with col4:
                    st.metric("Format", image.format or "Unknown")
                
            except Exception as e:
                st.error(f"❌ Không thể đọc ảnh: {str(e)}")
                return
            
            # Nút dự đoán
            if st.button("🔍 Bắt Đầu Chẩn Đoán", type="primary", use_container_width=True):
                
                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    status_text.text("📤 Đang tải ảnh lên server...")
                    progress_bar.progress(25)
                    
                    # Reset file pointer
                    uploaded_file.seek(0)
                    
                    status_text.text("🤖 Đang chạy mô hình AI...")
                    progress_bar.progress(50)
                    
                    # Dự đoán
                    results = predict_image(uploaded_file, selected_model)
                    
                    progress_bar.progress(75)
                    status_text.text("📊 Đang xử lý kết quả...")
                    
                    if results:
                        progress_bar.progress(100)
                        status_text.text("✅ Hoàn thành!")
                        time.sleep(0.5)  # Pause để user thấy completion
                        
                        # Clear progress indicators
                        progress_bar.empty()
                        status_text.empty()
                        
                        st.success("🎉 Phân tích hoàn tất!")
                        display_prediction_results(results, image)
                    else:
                        progress_bar.empty()
                        status_text.empty()
                        st.error("❌ Có lỗi xảy ra trong quá trình dự đoán")
                        
                except Exception as e:
                    progress_bar.empty()
                    status_text.empty()
                    st.error(f"❌ Lỗi: {str(e)}")
    
    with tab2:
        st.header("📋 Thông Tin Các Loại Bệnh Da")
        
        disease_info = get_disease_info()
        if disease_info:
            display_disease_info(disease_info)
        else:
            st.error("❌ Không thể tải thông tin bệnh")
    
    with tab3:
        st.header("📊 Demo với Dataset Test")
        st.info("🚧 Tính năng này sẽ được phát triển trong phiên bản tiếp theo")
        
        # Hiển thị thống kê dataset
        st.subheader("📈 Thống Kê Dataset Test")
        
        dataset_stats = {
            'akiec': 10, 'bcc': 10, 'bkl': 10, 'df': 7,
            'mel': 10, 'nv': 10, 'vasc': 11
        }
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Biểu đồ pie
            fig_pie = px.pie(
                values=list(dataset_stats.values()),
                names=list(dataset_stats.keys()),
                title="Phân Bố Dataset Test"
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Biểu đồ bar
            fig_bar = px.bar(
                x=list(dataset_stats.keys()),
                y=list(dataset_stats.values()),
                title="Số Lượng Ảnh Theo Loại",
                labels={'x': 'Loại bệnh', 'y': 'Số lượng ảnh'}
            )
            st.plotly_chart(fig_bar, use_container_width=True)

if __name__ == "__main__":
    main()
