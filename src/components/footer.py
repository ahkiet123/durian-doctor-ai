"""
Component: Footer
"""
import streamlit as st


def render_footer():
    """Render footer của ứng dụng"""
    st.markdown(
    """
    <hr style="margin-top: 40px; border: 0; border-top: 1px solid #e0e0e0;">
    <div style='text-align: center; color: #666; font-family: sans-serif; padding: 20px 0;'>
        <p style='font-size: 16px; font-weight: 600; margin-bottom: 8px;'>
            🌳 Durian Doctor AI
        </p>
        <p style='font-size: 14px; margin-bottom: 8px;'>
            Hệ thống AI hỗ trợ chẩn đoán bệnh trên lá sầu riêng và tư vấn phác đồ điều trị phù hợp
        </p>
        <p style='font-size: 13px; margin-bottom: 12px;'>
            Phát triển bởi <b style='color: #333;'>Đặng Anh Kiệt</b> &copy; 2025
        </p>
        <p style='font-size: 12px; color: #999;'>
            <i> <b>MobileNetV2</b> • <b>Grad-CAM</b> • <b>Google Gemini</b> • <b>Streamlit</b></i>
        </p>
    </div>
    """,
    unsafe_allow_html=True
    )
