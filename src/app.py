"""
Durian Doctor - Ứng dụng AI chẩn đoán bệnh sầu riêng
Streamlit App với Grad-CAM, RAG (Local ChromaDB) và Gemini Chatbot
"""

import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import google.generativeai as genai
import os
from dotenv import load_dotenv

# --- IMPORT MỚI CHO RAG ---
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Chroma

# Load biến môi trường từ file .env
load_dotenv()

# --- CẤU HÌNH TRANG ---
st.set_page_config(
    page_title="Durian Doctor", 
    page_icon="🌳", 
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- 1. SETUP GEMINI API ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

# --- 2. CẤU HÌNH RAG (LOCAL EMBEDDINGS) ---
class LocalSentenceEmbeddings:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        embs = self.model.encode(texts, show_progress_bar=False)
        return [emb.tolist() if hasattr(emb, 'tolist') else list(emb) for emb in embs]

    def embed_query(self, text):
        emb = self.model.encode([text], show_progress_bar=False)[0]
        return emb.tolist() if hasattr(emb, 'tolist') else list(emb)

def build_chroma_db_if_missing(kb_path: str, db_path: str):
    """Build Chroma DB từ file knowledge base nếu chưa tồn tại"""
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    
    # Đọc file knowledge base
    if not os.path.exists(kb_path):
        print(f"⚠️ Không tìm thấy file knowledge base: {kb_path}")
        return None
    
    with open(kb_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Chia nhỏ văn bản
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " "]
    )
    chunks = text_splitter.split_text(content)
    
    if not chunks:
        print("⚠️ Không có nội dung để tạo DB")
        return None
    
    print(f"📚 Đang tạo Chroma DB với {len(chunks)} đoạn văn bản...")
    
    # Tạo DB mới
    embedding_function = LocalSentenceEmbeddings()
    db = Chroma.from_texts(
        texts=chunks,
        embedding=embedding_function,
        persist_directory=db_path
    )
    print(f"✅ Đã tạo Chroma DB tại: {db_path}")
    return db

@st.cache_resource
def load_vector_db():
    """Load Vector Database, tự động build nếu chưa có (hỗ trợ Streamlit Cloud)"""
    try:
        embedding_function = LocalSentenceEmbeddings()
        base_dir = os.path.dirname(__file__)
        kb_path = os.path.join(base_dir, '..', 'knowledge_base', 'durian_diseases.txt')
        
        # Thử local path trước, nếu không ghi được thì dùng /tmp (Streamlit Cloud)
        local_db_path = os.path.join(base_dir, '..', 'knowledge_base', 'chroma_db')
        
        # Kiểm tra nếu local DB đã tồn tại và có dữ liệu
        if os.path.exists(local_db_path) and os.listdir(local_db_path):
            print("📂 Loading existing local Chroma DB...")
            db = Chroma(persist_directory=local_db_path, embedding_function=embedding_function)
            return db
        
        # Trên Streamlit Cloud: dùng /tmp (writable)
        import tempfile
        cloud_db_path = os.path.join(tempfile.gettempdir(), 'chroma_durian_db')
        
        # Nếu đã build trong /tmp rồi thì load
        if os.path.exists(cloud_db_path) and os.listdir(cloud_db_path):
            print("📂 Loading existing Chroma DB from /tmp...")
            db = Chroma(persist_directory=cloud_db_path, embedding_function=embedding_function)
            return db
        
        # Chưa có DB → build mới vào /tmp
        print("🔄 Chroma DB chưa tồn tại, đang tự động tạo trong /tmp...")
        db = build_chroma_db_if_missing(kb_path, cloud_db_path)
        return db
        
    except Exception as e:
        print(f"Lỗi load DB: {e}")
        import traceback
        traceback.print_exc()
        return None

# --- 3. CẤU HÌNH CLASS BỆNH (11 Lớp) ---
CLASS_NAMES = [
    'anthracnose_disease', 'canker_disease', 'fruit_rot', 'leaf_healthy',
    'mealybug_infestation', 'pink_disease', 'sooty_mold', 'stem_blight',
    'stem_cracking_gummosis', 'thrips_disease', 'yellow_leaf'
]

CLASS_NAMES_VI = {
    'anthracnose_disease': 'Bệnh thán thư (Anthracnose)',
    'canker_disease': 'Bệnh loét thân (Canker)',
    'fruit_rot': 'Thối trái (Fruit Rot)',
    'leaf_healthy': 'Lá khỏe mạnh (Healthy)',
    'mealybug_infestation': 'Rệp sáp (Mealybug)',
    'pink_disease': 'Bệnh hồng (Pink Disease)',
    'sooty_mold': 'Nấm muội đen (Sooty Mold)',
    'stem_blight': 'Cháy thân (Stem Blight)',
    'stem_cracking_gummosis': 'Nứt thân xì mủ (Gummosis)',
    'thrips_disease': 'Bọ trĩ (Thrips)',
    'yellow_leaf': 'Vàng lá (Yellow Leaf)'
}

# --- 4. HÀM LOAD MODEL VISION ---
@st.cache_resource
def load_model():
    """Load model MobileNetV2 đã train"""
    try:
        model = models.mobilenet_v2(weights=None)
        num_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, len(CLASS_NAMES))
        )
        
        model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'best_mobilenet_v2.pth')
        
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            model.eval()
            return model, True
        else:
            return None, False
    except Exception as e:
        return None, False

# --- 5. HÀM DỰ ĐOÁN & GRAD-CAM ---
def predict_and_gradcam(image, model):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    input_tensor = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        confidence, predicted_idx = torch.max(probabilities, 0)
    
    predicted_label = CLASS_NAMES[predicted_idx.item()]
    
    # Grad-CAM
    target_layers = [model.features[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)
    
    rgb_img = np.float32(image.resize((224, 224))) / 255
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)
    visualization = show_cam_on_image(rgb_img, grayscale_cam[0], use_rgb=True)
    
    top3_prob, top3_idx = torch.topk(probabilities, 3)
    top3 = [(CLASS_NAMES[idx.item()], prob.item()) for idx, prob in zip(top3_idx, top3_prob)]
    
    return predicted_label, confidence.item(), visualization, top3

# --- 6. GIAO DIỆN CHÍNH (UI) ---
def main():
    st.title("🌳 Durian Doctor AI")
    st.markdown("**Hệ thống AI chẩn đoán bệnh sầu riêng & Tư vấn điều trị**")
    st.markdown("---")
    
    # Load tài nguyên
    model, model_loaded = load_model()
    vector_db = load_vector_db()
    
    if not model_loaded:
        st.warning("⚠️ Chưa tìm thấy file model. Vui lòng train xong model.")
    if vector_db is None:
        st.warning("⚠️ Chưa tìm thấy Database. Chatbot sẽ không dùng RAG.")

    tab1, tab2 = st.tabs(["📷 Chẩn đoán bệnh", "💬 Hỏi đáp AI"])
    
    # === TAB 1: CHẨN ĐOÁN ===
    with tab1:
        st.subheader("📷 Tải ảnh lên để chẩn đoán")
        option = st.radio("Nguồn ảnh:", ("📁 Tải ảnh", "📸 Chụp ảnh"), horizontal=True)
        
        image = None
        if option == "📸 Chụp ảnh":
            camera_file = st.camera_input("Chụp ảnh")
            if camera_file: image = Image.open(camera_file).convert('RGB')
        else:
            uploaded_file = st.file_uploader("Chọn ảnh...", type=["jpg", "png", "jpeg"])
            if uploaded_file: image = Image.open(uploaded_file).convert('RGB')
        
        if image:
            col1, col2 = st.columns(2)
            with col1: st.image(image, caption="Ảnh gốc", use_container_width=True)
            
            if st.button("🔍 Chẩn đoán ngay", type="primary"):
                if model_loaded:
                    with st.spinner('🔄 Đang phân tích...'):
                        label, conf, heatmap, top3 = predict_and_gradcam(image, model)
                        
                        with col2: st.image(heatmap, caption="Heatmap vùng bệnh", use_container_width=True)
                        
                        st.markdown("---")
                        st.success(f"🎯 **{CLASS_NAMES_VI.get(label, label)}** (Độ tin cậy: {conf*100:.1f}%)")
                        
                        with st.expander("Xem chi tiết xác suất"):
                            for n, p in top3: st.write(f"- {CLASS_NAMES_VI.get(n, n)}: {p*100:.1f}%")
                        
                        # Lưu trạng thái để Chatbot biết
                        st.session_state['diagnosis_vi'] = CLASS_NAMES_VI.get(label, label)
                else:
                    st.error("Chưa load được model.")

    # === TAB 2: CHATBOT RAG (FINAL UPDATED) ===
    with tab2:
        st.subheader("💬 Hỏi đáp với Chuyên gia AI")
        
        if not GOOGLE_API_KEY:
            st.warning("⚠️ Chưa cấu hình API Key.")
        
        # Hiển thị kết quả chẩn đoán gần nhất
        if 'diagnosis_vi' in st.session_state:
            st.info(f"📋 Kết quả chẩn đoán gần nhất: **{st.session_state['diagnosis_vi']}**")
        
        # Chat History
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Input
        if prompt := st.chat_input("Hỏi về bệnh sầu riêng, cách điều trị..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"): st.markdown(prompt)
            
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                
                if not GOOGLE_API_KEY:
                    bot_reply = "⚠️ Thiếu API Key."
                else:
                    # 1. RAG: Tìm kiếm trong Vector DB
                    retrieved_block = ""
                    try:
                        if vector_db:
                            docs = vector_db.similarity_search(prompt, k=3)
                            if docs:
                                content_list = [f"[{i+1}] {d.page_content}" for i, d in enumerate(docs)]
                                retrieved_block = "THÔNG TIN THAM KHẢO TỪ TÀI LIỆU (RAG):\n" + "\n\n".join(content_list)
                            else:
                                retrieved_block = "Không tìm thấy thông tin liên quan trong tài liệu."
                    except Exception as e:
                        print(f"RAG Error: {e}")
                    
                    # 2. Context Chẩn đoán
                    diag_context = ""
                    if 'diagnosis_vi' in st.session_state:
                        diag_context = f"LƯU Ý NGỮ CẢNH: Người dùng vừa upload ảnh và được AI chẩn đoán cây bị bệnh: {st.session_state['diagnosis_vi']}."

                    # 3. Chat History Context (Tạo trí nhớ ngắn hạn)
                    chat_history_text = ""
                    # Lấy 6 tin nhắn gần nhất để làm ngữ cảnh (User - Bot - User - Bot...)
                    recent_msgs = st.session_state.messages[-6:]
                    for msg in recent_msgs:
                        role_label = "Người dùng" if msg["role"] == "user" else "Durian Doctor"
                        chat_history_text += f"{role_label}: {msg['content']}\n"

                    # 4. System Prompt (Cập nhật quy tắc nhớ & hỏi ngược)
                    system_prompt = """
Bạn là "Durian Doctor" - chuyên gia nông nghiệp hàng đầu về cây sầu riêng tại Việt Nam.

QUY TẮC CỐT LÕI (BẮT BUỘC):
1. **KIỂM TRA LỊCH SỬ CHAT (Context Awareness):** Trước khi hỏi lại người dùng, HÃY ĐỌC KỸ phần "LỊCH SỬ TRÒ CHUYỆN" bên dưới. Nếu người dùng đã cung cấp thông tin (như tuổi cây, giống, giai đoạn) ở các câu trước, **TUYỆT ĐỐI KHÔNG HỎI LẠI**. Hãy tự xâu chuỗi thông tin để trả lời.
2. **Tư vấn có tâm:** Nếu người dùng hỏi chung chung (VD: "Bón phân gì?"), hãy hỏi thêm 2-3 thông tin quan trọng nhất (Tuổi cây, Giai đoạn sinh trưởng, Tình trạng đất) để tư vấn chính xác.
3. **An toàn:** Chỉ đưa ra tên thuốc/liều lượng nếu có trong tài liệu. Không bịa số. Chỉ trả lời về sầu riêng.

CẤU TRÚC TRẢ LỜI:
- Chào hỏi ngắn gọn.
- Nếu thiếu thông tin -> Hỏi lại.
- Nếu đủ thông tin -> Đưa ra phác đồ chi tiết (Phân bón, Thuốc, Cách làm) dựa trên "THÔNG TIN THAM KHẢO".
                    """
                    
                    # 5. Build Full Prompt
                    full_prompt = f"""
{system_prompt}

{retrieved_block}

{diag_context}

LỊCH SỬ TRÒ CHUYỆN (CONTEXT):
{chat_history_text}

NGƯỜI DÙNG HỎI (CÂU MỚI NHẤT):
{prompt}
"""
                    
                    # 6. Call Gemini
                    try:
                        model_gemini = genai.GenerativeModel('gemini-2.0-flash')
                        response = model_gemini.generate_content(full_prompt)
                        bot_reply = response.text
                    except Exception as e:
                        bot_reply = f"⚠️ Lỗi kết nối Google Gemini: {e}"
                
                message_placeholder.markdown(bot_reply)
                st.session_state.messages.append({"role": "assistant", "content": bot_reply})
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
        🌳 Durian Doctor AI - Đồ án tốt nghiệp<br>
        Powered by MobileNetV2 + Grad-CAM + Google Gemini
        </div>
        """, 
        unsafe_allow_html=True
    )
if __name__ == "__main__":
    main()