"""
Durian Doctor - Cấu hình chung
"""
import streamlit as st
from groq import Groq
import os
from dotenv import load_dotenv

# Load biến môi trường
load_dotenv()

# Global Groq client
_groq_client = None


def setup_page():
    """Cấu hình trang Streamlit"""
    st.set_page_config(
        page_title="Durian Doctor AI",
        page_icon="🌳",
        layout="centered",
        initial_sidebar_state="expanded"
    )


def setup_gemini():
    """Setup Groq API (tương thích cả local và Streamlit Cloud)"""
    global _groq_client
    
    # Try Streamlit secrets first (for cloud deployment)
    api_key = ""
    try:
        api_key = st.secrets.get("GROQ_API_KEY", "")
    except:
        # Fallback to .env (for local development)
        api_key = os.getenv("GROQ_API_KEY", "")
    
    if api_key and not _groq_client:
        _groq_client = Groq(api_key=api_key)
    return api_key


def get_gemini_client():
    """Lấy Groq client instance (tên giữ nguyên để tương thích)"""
    if not _groq_client:
        setup_gemini()
    return _groq_client


def get_gemini_model_name():
    """Trả về tên model Groq phù hợp
    
    Returns:
        str: Model name cho Groq API
    """
    # Llama 3.3 70B - Balance tốt giữa speed & quality
    # Alternatives: llama-3.1-8b-instant (nhanh hơn), mixtral-8x7b-32768 (context lớn)
    return 'llama-3.3-70b-versatile'


