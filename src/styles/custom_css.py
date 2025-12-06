"""
Custom CSS styles cho Durian Doctor
"""
import streamlit as st


def inject_custom_css():
    """Inject CSS để Việt hóa và style carousel"""
    st.markdown("""
    <style>
        /* Ẩn "Drag and drop file here" */
        [data-testid="stFileUploaderDropzoneInstructions"] > span:nth-child(2) {
            font-size: 0 !important;
        }
        [data-testid="stFileUploaderDropzoneInstructions"] > span:nth-child(2)::after {
            content: "Kéo thả ảnh vào đây";
            font-size: 14px !important;
            color: #31333F;
        }
        
        /* Ẩn "Limit 200MB per file • JPG, PNG, JPEG" */
        [data-testid="stFileUploaderDropzoneInstructions"] > span:nth-child(3) {
            font-size: 0 !important;
        }
        [data-testid="stFileUploaderDropzoneInstructions"] > span:nth-child(3)::after {
            content: "Định dạng: JPG, PNG, JPEG";
            font-size: 12px !important;
            color: #808495;
        }
        
        /* Việt hóa nút Browse files */
        [data-testid="stFileUploaderDropzone"] button[kind="secondary"] {
            font-size: 0 !important;
        }
        [data-testid="stFileUploaderDropzone"] button[kind="secondary"]::after {
            content: "Chọn tệp";
            font-size: 14px !important;
        }
        
        /* Carousel trong dialog */
        iframe[title="streamlit_carousel.streamlit_carousel"] {
            height: 180px !important;
            margin-bottom: -20px !important;
        }
    </style>
    """, unsafe_allow_html=True)
