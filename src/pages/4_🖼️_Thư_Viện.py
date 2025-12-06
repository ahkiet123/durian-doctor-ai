"""
Page 4: Thư viện ảnh mẫu (Coming Soon)
"""
import streamlit as st
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from components.footer import render_footer

st.header("🖼️ Thư viện ảnh mẫu")

st.info("🚧 **Tính năng đang phát triển**\n\nChức năng này sẽ cho phép bạn:\n- Xem ảnh mẫu các loại bệnh trên sầu riêng\n- So sánh triệu chứng với ảnh của bạn\n- Tải ảnh mẫu để test hệ thống")

st.markdown("---")
st.caption("Dự kiến ra mắt trong phiên bản tiếp theo")

render_footer()
