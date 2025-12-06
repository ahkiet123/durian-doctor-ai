"""
Component: Hướng dẫn chụp ảnh với Carousel
"""
import streamlit as st
from streamlit_carousel import carousel


# Dữ liệu slides hướng dẫn
GUIDE_ITEMS = [
    dict(title="", text="", img="https://placehold.co/300x180/2E7D32/white?text=1.+Cận+cảnh"),
    dict(title="", text="", img="https://placehold.co/300x180/1565C0/white?text=2.+Đủ+sáng"),
    dict(title="", text="", img="https://placehold.co/300x180/6A1B9A/white?text=3.+Rõ+nét"),
    dict(title="", text="", img="https://placehold.co/300x180/C62828/white?text=4.+Chụp+xa"),
    dict(title="", text="", img="https://placehold.co/300x180/E65100/white?text=5.+Ngược+sáng"),
]

GUIDE_CAPTIONS = [
    ("✅ Chụp cận cảnh (20-50cm)", "Để vùng bệnh chiếm hơn 50% khung hình"),
    ("✅ Đủ ánh sáng", "Chụp ngoài trời, tránh bóng râm"),
    ("✅ Giữ camera ổn định", "Ảnh không bị mờ, rung hoặc nhòe"),
    ("❌ TRÁNH: Chụp quá xa", "Không chụp toàn cảnh cả cây"),
    ("❌ TRÁNH: Ngược sáng", "Không chụp ngược sáng hoặc bóng tối"),
]


def render_photo_guide():
    """Render nút mở dialog hướng dẫn chụp ảnh"""
    
    @st.dialog("📷 Hướng dẫn chụp ảnh", width="small")
    def show_carousel_guide():
        carousel(items=GUIDE_ITEMS, width=1)
        for title, desc in GUIDE_CAPTIONS:
            st.markdown(f"**{title}**: {desc}")
    
    if st.button("📌 Xem hướng dẫn chụp ảnh để AI phân tích chính xác", type="tertiary"):
        show_carousel_guide()
