# Durian Disease Detection - Deployment Guide

Deployed at: [Your Streamlit App URL]

## 🚀 Quick Deploy to Streamlit Cloud

### 1. Secrets Configuration

Add to Streamlit Cloud Secrets (TOML format):

```toml
GROQ_API_KEY = "your_groq_api_key_here"
```

### 2. App Configuration

- **Repository**: your-github-username/durian-doctor-ai
- **Branch**: main
- **Main file**: src/app.py

### 3. Environment Variables

The app automatically detects:
- **Local**: Uses `.env` file
- **Cloud**: Uses Streamlit secrets

## 📝 Notes

- API Key is stored securely in Streamlit Cloud
- Free tier: 1GB RAM, sufficient for this app
- Model: Groq Llama 3.3 70B (14,400 requests/day free)

## 🔧 Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
python -m streamlit run src/app.py
```

## 📊 Features

- ✅ Disease detection from images
- ✅ AI chatbot with RAG
- ✅ Explainable AI (Grad-CAM)
- ✅ Treatment recommendations
