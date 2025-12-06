"""
Durian Doctor - Cấu hình chung
"""
import streamlit as st
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load biến môi trường
load_dotenv()


def setup_page():
    """Cấu hình trang Streamlit"""
    st.set_page_config(
        page_title="Durian Doctor AI",
        page_icon="🌳",
        layout="centered",
        initial_sidebar_state="expanded"
    )


def setup_gemini():
    """Setup Google Gemini API"""
    api_key = os.getenv("GOOGLE_API_KEY", "")
    if api_key:
        genai.configure(api_key=api_key)
    return api_key


def get_gemini_model():
    """Lấy Gemini model instance"""
    return genai.GenerativeModel('gemini-2.0-flash')
