# 🌳 Durian Doctor AI

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Gemini](https://img.shields.io/badge/Gemini-8E75B2?style=for-the-badge&logo=googlebard&logoColor=white)

> **Tiểu luận tốt nghiệp - Hệ thống AI chẩn đoán và tư vấn bệnh trên cây sầu riêng**  
> **Dev:** Đặng Anh Kiệt

---

## 🚀 Demo Trực Tuyến

👉 **[Durian Doctor AI - Live App](https://durian-doctor-ai.streamlit.app/)**

---

## ✨ Tính năng

| Tính năng | Mô tả |
|-----------|-------|
| 📸 **Chẩn đoán bệnh** | Phân loại 11 loại bệnh từ ảnh (MobileNetV2 + Grad-CAM) |
| 💬 **Chatbot RAG** | Tư vấn bằng AI với dữ liệu chuyên sâu (Gemini + ChromaDB) |
| 📷 **Hướng dẫn chụp ảnh** | Carousel hướng dẫn chụp ảnh đúng cách |
| 🧠 **Quá trình suy nghĩ** | Hiển thị chi tiết AI đang xử lý gì |

---

## 📁 Cấu trúc Project

```
├── src/
│   ├── app.py                 # Entry point
│   ├── config.py              # Cấu hình chung
│   ├── pages/                 # Multi-page Streamlit
│   │   ├── 1_📷_Chẩn_Đoán.py
│   │   ├── 2_💬_Hỏi_Đáp.py
│   │   ├── 3_📊_Lịch_Sử.py
│   │   └── 4_🖼️_Thư_Viện.py
│   ├── model/                 # AI Model
│   │   ├── classifier.py      # Load & predict
│   │   └── gradcam.py         # Explainable AI
│   ├── components/            # UI Components
│   ├── styles/                # CSS
│   └── prompts/               # LLM Prompts
├── models/                    # Trained weights
├── knowledge_base/            # RAG documents
└── tests/                     # Unit tests
```

---

## 🛠️ Cài đặt

```bash
# Clone
git clone https://github.com/ahkiet123/durian-doctor-ai.git
cd durian-doctor-ai

# Install
pip install -r requirements.txt

# Config (.env)
GOOGLE_API_KEY=your_api_key_here

# Run
streamlit run src/app.py
```

---

## 👨‍💻 Tác giả

**Đặng Anh Kiệt** | [GitHub](https://github.com/ahkiet123)

© 2025 - Tiểu luận tốt nghiệp
