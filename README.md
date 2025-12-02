# Durian Doctor AI — Hệ thống AI chẩn đoán bệnh sầu riêng

truy cập bản demo (chưa có model nhận diện qua ảnh chỉ có thể thử qua bot chat AI chuyên viên sầu riêng. ) qua đường link:
durian-doctor-ai.streamlit.app
Durian Doctor AI là một hệ thống bằng Python để phát hiện và hỗ trợ chẩn đoán các bệnh thường gặp trên cây sầu riêng (lá, thân, trái). Dự án tích hợp pipeline xử lý ảnh, mô-đun huấn luyện/đánh giá mô hình PyTorch, chức năng giải thích kết quả (Grad-CAM), và ứng dụng giao diện người dùng bằng Streamlit cùng một knowledge base để hỗ trợ mô-đun RAG (retrieval-augmented generation).

Phiên bản README này được soạn dựa trên cấu trúc và nội dung hiện tại của repository (các file trong src/, notebooks/, knowledge_base/, requirements.txt, v.v.).

---

## Mục lục
- [Tổng quan](#tổng-quan)
- [Tiến độ hiện tại (dựa trên nội dung repo)](#tiến-độ-hiện-tại-dựa-trên-nội-dung-repo)
- [Tính năng chính](#tính-năng-chính)
- [Kiến trúc & Công nghệ chính](#kiến-trúc--công-nghệ-chính)
- [Yêu cầu & Cài đặt nhanh](#yêu-cầu--cài-đặt-nhanh)
- [Chạy nhanh (Quick start)](#chạy-nhanh-quick-start)
- [Cấu trúc repository (tổng quan)](#cấu-trúc-repository-tổng-quan)
- [Hướng dẫn dùng các script chính](#hướng-dẫn-dùng-các-script-chính)
- [Đánh giá & Chỉ số theo dõi](#đánh-giá--chỉ-số-theo-dõi)
- [Đóng góp & Giấy phép](#đóng-góp--giấy-phép)
- [Thông tin liên hệ](#thông-tin-liên-hệ)

---

## Tổng quan
Mục tiêu: xây dựng giải pháp All-in-One để:
- Phân loại/chẩn đoán bệnh trên ảnh lá/thân/trái sầu riêng.
- Trực quan hoá vùng ảnh ảnh hưởng (Grad-CAM).
- Hỗ trợ tư vấn (chatbot RAG sử dụng knowledge base).
- Cung cấp giao diện Streamlit để upload ảnh, xem kết quả và hỏi đáp.

---

## Tiến độ hiện tại (dựa trên nội dung repo)
Tóm tắt những phần đã có trong repository và trạng thái hiện tại dựa trên file code và tài liệu:

Hoàn thành / Có sẵn trong repo
- Tiền xử lý ảnh: có notebook `notebooks/01_preprocess_resize.py` và file README/notes cho pipeline resize (đã xử lý ảnh về 224x224 theo tài liệu).
- Loader dataset: `src/dataset.py` — class/logic để load dữ liệu cho huấn luyện (PyTorch).
- Huấn luyện thử nghiệm: `src/train.py` — script huấn luyện (prototype có sẵn).
- Đánh giá: `src/evaluate.py` — script để chạy đánh giá trên tập test.
- Grad-CAM / giải thích: `src/gradcam_utils.py` — utilities để sinh heatmap giải thích.
- Streamlit app: `src/app.py` — scaffold ứng dụng UI (upload ảnh, dự đoán, tab Chatbot).
- Build knowledge base: `src/build_db.py` và `knowledge_base/durian_diseases.txt` (tập nội dung tri thức cho RAG).
- Yêu cầu môi trường: `requirements.txt` (liệt kê các thư viện cần thiết: PyTorch, torchvision, streamlit, chromadb, grad-cam, v.v.).
- .gitignore đã cấu hình để loại trừ data/models/outputs và file .env.

Phần đang chờ / khuyến nghị (xuất hiện trong tài liệu nội bộ)
- Huấn luyện chính thức trên toàn bộ dataset để tạo checkpoint sản phẩm (ví dụ: `models/best_mobilenet_v2.pth`) — hiện repo có script nhưng model checkpoint lớn chưa có.
- Tách module RAG thành file riêng (gợi ý: `src/rag.py`) để tách responsibility, hiện logic retrieval hoạt động inline trong `src/app.py`.
- Hoàn thiện hiển thị passages / nguồn tham chiếu trên giao diện chatbot để tăng minh bạch.
- (Tài liệu đề xuất) Kiểm thử mô hình trên tập test độc lập và lưu báo cáo đo lường.

Ghi chú: trạng thái trên được rút ra trực tiếp từ nội dung hiện có trong repo (các file trong `src/`, `notebooks/` và `knowledge_base/`) và tài liệu mô tả có trong repository.

---

## Tính năng chính
- Phân loại 10 lớp bệnh/healthy (ghi chú: danh sách lớp và nguồn dataset được nêu trong tài liệu luận án).
- Grad-CAM để hiển thị vùng ảnh mô hình quan tâm.
- Knowledge base + Chroma để lưu passages hỗ trợ RAG.
- Streamlit UI: upload ảnh → dự đoán → hiển thị heatmap + tab Chatbot để tra cứu cách điều trị/gợi ý.
- Scripts để build DB, huấn luyện, đánh giá, và inference.

---

## Kiến trúc & Công nghệ chính
- Ngôn ngữ: Python.
- Framework học sâu: PyTorch (repo dùng PyTorch theo requirements và mô tả).
- UI: Streamlit (`src/app.py`).
- Vector DB / RAG: ChromaDB + embeddings (logic build DB có trong `src/build_db.py`).
- Explainability: pytorch-grad-cam / grad-cam utilities (`src/gradcam_utils.py`).

---

## Yêu cầu & Cài đặt nhanh
- Python >= 3.8 (khuyến nghị 3.10+).
- GPU + CUDA nếu huấn luyện (yêu cầu tuỳ theo phiên bản torch trong `requirements.txt`).
- Cài đặt:
  1. Tạo virtualenv và kích hoạt:
     - python -m venv venv
     - source venv/bin/activate  (Linux/macOS) hoặc venv\Scripts\activate (Windows)
  2. Cài đặt dependencies:
     - pip install -r requirements.txt

Lưu ý: file `requirements.txt` trong repo có danh sách dài thư viện, bao gồm cả torch + torchvision (phiên bản +CUDA), chromadb, streamlit, grad-cam, v.v. Kiểm tra và chỉnh phiên bản phù hợp với hệ thống trước khi cài.

---

## Chạy nhanh (Quick start)
Các lệnh thường dùng (đã test theo cấu trúc repo):

- Xây dựng knowledge base (mã có sẵn):
  - python src/build_db.py

- Huấn luyện prototype / khởi chạy quá trình huấn luyện:
  - python src/train.py --config configs/train.yaml
  (nếu repo không có configs, kiểm tra tham số trong train.py)

- Đánh giá model:
  - python src/evaluate.py --checkpoint models/best_mobilenet_v2.pth

- Chạy giao diện Streamlit:
  - python -m streamlit run src/app.py

---

## Cấu trúc repository (những tệp & thư mục quan trọng hiện có)
- .gitignore
- requirements.txt
- # TIỂU LUẬN TỐT NGHIỆP HỆ THỐNG AI CHẨN.txt  (file luận văn/tài liệu mô tả chi tiết project)
- knowledge_base/
  - durian_diseases.txt
  - (đề xuất) chroma_db/ (bị ignore nhưng build script tạo)
- notebooks/
  - 01_preprocess_resize.py
- src/
  - app.py
  - build_db.py
  - dataset.py
  - evaluate.py
  - gradcam_utils.py
  - train.py
- (models/ và data/ được .gitignore — không có file model lớn trong repo)

---

## Hướng dẫn dùng các script chính (tóm tắt)
- src/build_db.py
  - Chuyển nội dung text trong `knowledge_base/durian_diseases.txt` thành embeddings và lưu collection Chroma cục bộ.
- src/dataset.py
  - Class/logic load ảnh đã resize và labels cho PyTorch DataLoader.
- src/train.py
  - Script huấn luyện: load dataset, backbone MobileNetV2 (theo luận án), optimizer AdamW, criterion CrossEntropyLoss. Kiểm tra/thiết lập seed, lưu checkpoint.
- src/evaluate.py
  - Tạo metrics (accuracy, precision/recall/f1), confusion matrix và lưu báo cáo.
- src/gradcam_utils.py
  - Tạo heatmap Grad-CAM để overlay lên ảnh gốc.
- src/app.py
  - Streamlit app: upload ảnh, predict, show heatmap, và tab Chatbot dùng retrieval từ Chroma DB để ghép prompt gửi LLM.

---

## Đánh giá & Chỉ số theo dõi
Nên lưu và theo dõi:
- Accuracy / Precision / Recall / F1 per class
- Confusion matrix
- Thời gian inference trung bình & memory footprint (quan trọng khi deploy offline)
- Versioning của model/checkpoint và dữ liệu (ví dụ: train_v1, train_v2...)

---

## Đóng góp
- Mở issue để báo lỗi hoặc đề xuất tính năng.
- Gửi pull request với mô tả thay đổi chi tiết và hướng dẫn thử nghiệm.
- Nếu muốn đóng góp code: làm việc trên branch feature/XXX → PR → review.

---

## Giấy phép
- Repository hiện chưa hiển thị file LICENSE. Nếu muốn công khai bản quyền, thêm file LICENSE (MIT / Apache-2.0 / GPL tuỳ yêu cầu).

---

## Thông tin liên hệ
- Chủ dự án / owner: ahkiet123
- Repo: https://github.com/ahkiet123/durian-doctor-ai

---

Ghi chú ngắn: README này được soạn dựa trên nội dung thực tế hiện có trong repository (các file trong thư mục src/, notebooks/, knowledge_base/ và requirements.txt). Các script chính đã có sẵn; checkpoint mô hình lớn và dữ liệu huấn luyện thường được giữ ngoài git (bị .gitignore).  
