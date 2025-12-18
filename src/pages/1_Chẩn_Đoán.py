"""
Page 1: Chẩn đoán bệnh sầu riêng
"""
import streamlit as st
from PIL import Image
import sys
import os

# Thêm path để import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from styles.custom_css import inject_custom_css
from components.photo_guide import render_photo_guide
from components.footer import render_footer
from model import load_model, predict_and_gradcam, CLASS_NAMES_VI

# Inject CSS
inject_custom_css()

st.header("📷 Chẩn đoán bệnh sầu riêng")

# Load model
model, model_loaded = load_model()

if not model_loaded:
    st.info("ℹ️ **Chức năng chẩn đoán ảnh chưa sẵn sàng**  \nModel AI đang được huấn luyện. Vui lòng sử dụng tab **Hỏi đáp AI** để tư vấn.")
    st.markdown("---")

# Hướng dẫn chụp ảnh
render_photo_guide()

# Upload ảnh
option = st.radio("Nguồn ảnh:", ("📁 Tải ảnh", "📸 Chụp ảnh"), horizontal=True)

image = None
if option == "📸 Chụp ảnh":
    camera_file = st.camera_input("Chụp ảnh")
    if camera_file:
        image = Image.open(camera_file).convert('RGB')
else:
    uploaded_file = st.file_uploader("Chọn ảnh...", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')

# Xử lý chẩn đoán
if image:
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Ảnh gốc", use_container_width=True)
    
    if st.button("🔍 Chẩn đoán ngay", type="primary"):
        if model_loaded:
            with st.spinner('🔄 Đang phân tích...'):
                label, conf, heatmap, top3 = predict_and_gradcam(image, model)
                
                with col2:
                    st.image(heatmap, caption="Heatmap vùng bệnh", use_container_width=True)
                
                st.markdown("---")
                st.success(f"🎯 **{CLASS_NAMES_VI.get(label, label)}** (Độ tin cậy: {conf*100:.1f}%)")
                
                with st.expander("Xem chi tiết xác suất"):
                    for n, p in top3:
                        st.write(f"- {CLASS_NAMES_VI.get(n, n)}: {p*100:.1f}%")
                
                # Lưu trạng thái để Chatbot biết
                st.session_state['diagnosis_vi'] = CLASS_NAMES_VI.get(label, label)
        else:
            st.error("Chưa load được model.")

# Footer
render_footer()
