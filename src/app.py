"""
Durian Doctor AI - Entry Point
Hệ thống AI chẩn đoán bệnh sầu riêng & Tư vấn điều trị
"""
import streamlit as st
from config import setup_page, setup_gemini
from styles.custom_css import inject_custom_css
from components.footer import render_footer


def main():
    # Setup
    setup_page()
    inject_custom_css()
    setup_gemini()
    
    # Header
    st.title("🌳 Durian Doctor AI")
    st.markdown("**Hệ thống AI hỗ trợ chẩn đoán bệnh sầu riêng**")
    st.markdown("---")
    
    # Welcome content
    st.markdown("""
    ### 👋 Chào mừng đến với Durian Doctor AI!
    
    Hệ thống AI thông minh giúp bạn:
    - 📷 **Chẩn đoán bệnh** từ ảnh chụp lá, thân, trái sầu riêng
    - 💬 **Tư vấn điều trị** với chatbot AI chuyên gia
    - 📊 **Theo dõi lịch sử** chẩn đoán (sắp ra mắt)
    - 🖼️ **Thư viện ảnh mẫu** các loại bệnh (sắp ra mắt)
    
    ---
    
    👈 **Chọn chức năng từ sidebar** để bắt đầu!
    """)
    
    # Quick stats (nếu có)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🦠 Loại bệnh nhận diện", "11")
    with col2:
        st.metric("🤖 Model AI", "MobileNetV2")
    with col3:
        st.metric("💬 Chatbot", "Gemini 2.0")
    
    # Footer
    render_footer()


if __name__ == "__main__":
    main()