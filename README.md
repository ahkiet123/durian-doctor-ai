# 🌳 Durian Doctor AI

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Gemini](https://img.shields.io/badge/Gemini-8E75B2?style=for-the-badge&logo=googlebard&logoColor=white)

> **Tiểu luận tốt nghiệp - Hệ thống AI chẩn đoán và tư vấn bệnh trên cây sầu riêng**  
> **Dev:** Đặng Anh Kiệt

---

## 🚀 Demo Trực Tuyến
Trải nghiệm ngay ứng dụng tại đây:

👉 **[Durian Doctor AI - Live App](https://durian-doctor-ai.streamlit.app/)**

(Lưu ý: Tính năng chẩn đoán hình ảnh đang được phát triển, hiện tại chưa thể dùng được. Chatbot đã hoạt động đầy đủ với dữ liệu chuyên sâu)

---

## 📖 Giới thiệu

**Durian Doctor AI** là giải pháp công nghệ hỗ trợ nông dân và chuyên gia trong việc chăm sóc cây sầu riêng. Hệ thống kết hợp **Thị giác máy tính (Computer Vision)** để chẩn đoán bệnh qua ảnh và **Generative AI** để tư vấn cách điều trị.

### ✨ Tính năng nổi bật

*   📸 **Chẩn đoán bệnh:** Phân loại bệnh trên lá, thân, trái sầu riêng (MobileNetV2).
*   🔍 **XAI:** Hiển thị vùng bệnh trên ảnh bằng kỹ thuật Grad-CAM.
*   💬 **Chatbot Chuyên gia:** Trả lời câu hỏi, tư vấn thuốc và phác đồ điều trị dựa trên tài liệu chuẩn (RAG + Gemini).
*   📚 **Minh bạch:** Trích dẫn nguồn tài liệu tham khảo cho từng câu trả lời.

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

## 🛠️ Cài đặt & Chạy Local

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
