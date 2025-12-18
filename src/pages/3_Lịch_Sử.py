"""
Page 3: Lịch sử chẩn đoán (Coming Soon)
"""
import streamlit as st
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from components.footer import render_footer

st.header("📊 Lịch sử chẩn đoán")

st.info("🚧 **Tính năng đang phát triển**\n\nChức năng này sẽ cho phép bạn:\n- Xem lại các kết quả chẩn đoán trước đó\n- Theo dõi tiến triển bệnh theo thời gian\n- Xuất báo cáo PDF")

st.markdown("---")
st.caption("Dự kiến ra mắt trong phiên bản tiếp theo")

render_footer()
