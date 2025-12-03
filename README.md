# 🌳 Durian Doctor AI

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Gemini](https://img.shields.io/badge/Gemini-8E75B2?style=for-the-badge&logo=googlebard&logoColor=white)

> **Tiểu luận tốt nghiệp - Hệ thống AI chẩn đoán và tư vấn bệnh sầu riêng**  
> **Tác giả:** Đặng Anh Kiệt

---

## 🚀 Demo Trực Tuyến

Trải nghiệm ngay ứng dụng tại đây:  
👉 **[Durian Doctor AI - Live App](https://durian-doctor-ai.streamlit.app/)**

*(Lưu ý: Tính năng chẩn đoán hình ảnh đang sử dụng mô hình demo, Chatbot hoạt động đầy đủ với dữ liệu chuyên sâu)*

---

## 📖 Giới thiệu

**Durian Doctor AI** là giải pháp công nghệ hỗ trợ nông dân và chuyên gia trong việc chăm sóc cây sầu riêng. Hệ thống kết hợp **Thị giác máy tính (Computer Vision)** để chẩn đoán bệnh qua ảnh và **AI tạo sinh (Generative AI)** để tư vấn cách điều trị.

### ✨ Tính năng nổi bật

*   📸 **Chẩn đoán bệnh:** Phân loại bệnh trên lá, thân, trái sầu riêng (MobileNetV2).
*   🔍 **Giải thích AI:** Hiển thị vùng bệnh trên ảnh bằng kỹ thuật Grad-CAM.
*   💬 **Chatbot Chuyên gia:** Trả lời câu hỏi, tư vấn thuốc và phác đồ điều trị dựa trên tài liệu chuẩn (RAG + Gemini).
*   📚 **Minh bạch:** Trích dẫn nguồn tài liệu tham khảo cho từng câu trả lời.

---

## 🛠️ Cài đặt & Chạy Local

1.  **Clone dự án:**
    ```bash
    git clone https://github.com/ahkiet123/durian-doctor-ai.git
    cd durian-doctor-ai
    ```

2.  **Cài đặt thư viện:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Cấu hình:**
    *   Tạo file `.env` và thêm API Key của Gemini:
        ```env
        GOOGLE_API_KEY=your_api_key_here
        ```

4.  **Chạy ứng dụng:**
    ```bash
    streamlit run src/app.py
    ```

---

## 👨‍💻 Tác giả

*   **Họ tên:** Đặng Anh Kiệt
*   **Dự án:** Tiểu luận tốt nghiệp
*   **Liên hệ:** [GitHub Profile](https://github.com/ahkiet123)

---
*Made with ❤️ for Vietnam Agriculture*
