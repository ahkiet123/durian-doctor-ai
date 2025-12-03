"""
Durian Doctor - Ứng dụng AI chẩn đoán bệnh sầu riêng
Streamlit App với Grad-CAM, RAG (Local ChromaDB) và Gemini Chatbot
"""

import streamlit as st
from PIL import Image
import google.generativeai as genai
import os
from dotenv import load_dotenv

# --- IMPORT MODULES ---
from rag_engine import load_vector_db
from model_utils import load_model, predict_and_gradcam, CLASS_NAMES_VI

# Load biến môi trường từ file .env
load_dotenv()

# --- CẤU HÌNH TRANG ---
st.set_page_config(
    page_title="Durian Doctor", 
    page_icon="🌳", 
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- SETUP GEMINI API ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

# --- GIAO DIỆN CHÍNH (UI) ---
def main():
    st.title("🌳 Durian Doctor AI")
    st.markdown("**Hệ thống AI chẩn đoán bệnh sầu riêng & Tư vấn điều trị**")
    st.markdown("---")
    
    # Load tài nguyên (không hiển thị warning ở đây)
    model, model_loaded = load_model()
    vector_db = load_vector_db()

    tab1, tab2 = st.tabs(["📷 Chẩn đoán bệnh", "💬 Hỏi đáp AI"])
    
    # === TAB 1: CHẨN ĐOÁN ===
    with tab1:
        # Hiển thị thông báo model trong tab này thôi
        if not model_loaded:
            st.info("ℹ️ **Chức năng chẩn đoán ảnh chưa sẵn sàng**  \nModel AI đang được huấn luyện. Vui lòng sử dụng tab **Hỏi đáp AI** để tư vấn.")
            st.markdown("---")
        
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

    # === TAB 2: CHATBOT RAG ===
    with tab2:
        st.subheader("💬 Hỏi đáp với Chuyên gia AI")
        
        # Toggle hiển thị quá trình suy nghĩ
        show_thinking = st.toggle("🧠 Hiển thị quá trình suy nghĩ", value=False, help="Xem AI đang làm gì")
        
        # Hiển thị kết quả chẩn đoán gần nhất (nếu có)
        if 'diagnosis_vi' in st.session_state:
            st.info(f"📋 Kết quả chẩn đoán gần nhất: **{st.session_state['diagnosis_vi']}**")
        
        # Chat History
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Container cho messages với chiều cao cố định để input luôn ở dưới
        chat_container = st.container(height=450)
        
        # Hiển thị messages trong container
        with chat_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
        
        # Input luôn ở dưới cùng
        prompt = st.chat_input("Hỏi về bệnh sầu riêng, cách điều trị...")
        
        if prompt:
            # Thêm message user vào history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Hiển thị trong container
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(prompt)
            
                # Xử lý và hiển thị response
                with st.chat_message("assistant"):
                    if not GOOGLE_API_KEY:
                        st.warning("⚠️ Vui lòng cấu hình API Key trong phần Settings.")
                    else:
                        # Container cho thinking process
                        thinking_container = st.empty()
                    
                    # === STEP 1: Tìm kiếm RAG ===
                    if show_thinking:
                        with thinking_container.container():
                            st.markdown("🔍 **Đang tìm kiếm trong cơ sở tri thức...**")
                            with st.status("Truy vấn RAG Database", expanded=True) as status:
                                st.write("📚 Kết nối ChromaDB...")
                    
                    retrieved_block = ""
                    retrieved_docs_display = []
                    try:
                        if vector_db:
                            docs = vector_db.similarity_search(prompt, k=3)
                            if docs:
                                for i, d in enumerate(docs):
                                    retrieved_docs_display.append(f"**[{i+1}]** {d.page_content[:150]}...")
                                content_list = [f"[{i+1}] {d.page_content}" for i, d in enumerate(docs)]
                                retrieved_block = "THÔNG TIN THAM KHẢO TỪ TÀI LIỆU (RAG):\n" + "\n\n".join(content_list)
                    except Exception as e:
                        print(f"RAG Error: {e}")
                    
                    if show_thinking:
                        with thinking_container.container():
                            with st.status("Truy vấn RAG Database", expanded=True, state="complete") as status:
                                if retrieved_docs_display:
                                    st.write("✅ Tìm thấy tài liệu liên quan:")
                                    for doc in retrieved_docs_display:
                                        st.caption(doc)
                                else:
                                    st.write("ℹ️ Không tìm thấy tài liệu cụ thể")
                    
                    # === STEP 2: Chuẩn bị context ===
                    if show_thinking:
                        with thinking_container.container():
                            with st.status("Truy vấn RAG Database", expanded=False, state="complete"):
                                st.write("✅ Hoàn tất")
                            with st.status("Xây dựng ngữ cảnh", expanded=True) as status:
                                st.write("📝 Phân tích lịch sử hội thoại...")
                    
                    diag_context = ""
                    if 'diagnosis_vi' in st.session_state:
                        diag_context = f"LƯU Ý NGỮ CẢNH: Người dùng vừa upload ảnh và được AI chẩn đoán cây bị bệnh: {st.session_state['diagnosis_vi']}."

                    chat_history_text = ""
                    recent_msgs = st.session_state.messages[-6:]
                    for msg in recent_msgs:
                        role_label = "Người dùng" if msg["role"] == "user" else "Durian Doctor"
                        chat_history_text += f"{role_label}: {msg['content']}\n"

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
                    
                    full_prompt = f"""
{system_prompt}

{retrieved_block}

{diag_context}

LỊCH SỬ TRÒ CHUYỆN (CONTEXT):
{chat_history_text}

NGƯỜI DÙNG HỎI (CÂU MỚI NHẤT):
{prompt}
"""
                    
                    # === STEP 3: Gọi Gemini ===
                    if show_thinking:
                        with thinking_container.container():
                            with st.status("Truy vấn RAG Database", expanded=False, state="complete"):
                                st.write("✅ Hoàn tất")
                            with st.status("Xây dựng ngữ cảnh", expanded=False, state="complete"):
                                st.write("✅ Hoàn tất")
                            with st.status("🤖 Gemini đang suy nghĩ...", expanded=True) as status:
                                st.write("💭 Phân tích câu hỏi và tài liệu...")
                    
                    try:
                        model_gemini = genai.GenerativeModel('gemini-2.0-flash')
                        response = model_gemini.generate_content(full_prompt)
                        bot_reply = response.text
                    except Exception as e:
                        bot_reply = f"⚠️ Lỗi kết nối Google Gemini: {e}"
                    
                    # === Hoàn tất - Hiển thị kết quả ===
                    if show_thinking:
                        with thinking_container.container():
                            with st.status("Truy vấn RAG Database", expanded=False, state="complete"):
                                st.write("✅ Hoàn tất")
                            with st.status("Xây dựng ngữ cảnh", expanded=False, state="complete"):
                                st.write("✅ Hoàn tất")
                            with st.status("🤖 Gemini đang suy nghĩ...", expanded=False, state="complete"):
                                st.write("✅ Đã tạo câu trả lời")
                            st.markdown("---")
                    else:
                        thinking_container.empty()
                    
                    # Hiển thị response
                    st.markdown(bot_reply)
                    
                    # Hiển thị trích dẫn nguồn tài liệu (nếu có)
                    if retrieved_docs_display:
                        with st.expander("📚 Nguồn tài liệu tham khảo", expanded=False):
                            for i, doc in enumerate(retrieved_docs_display):
                                st.markdown(doc)
                                if i < len(retrieved_docs_display) - 1:
                                    st.divider()
                    
                    # Lưu message
                    st.session_state.messages.append({"role": "assistant", "content": bot_reply})

    # Footer
    st.markdown(
    """
    <hr style="margin-top: 40px; border: 0; border-top: 1px solid #e0e0e0;">
    <div style='text-align: center; color: #666; font-family: sans-serif; padding: 20px 0;'>
        <p style='font-size: 16px; font-weight: 600; margin-bottom: 8px;'>
            🌳 Durian Doctor AI
        </p>
        <p style='font-size: 14px; margin-bottom: 8px;'>
            Hệ thống AI hỗ trợ chẩn đoán bệnh sầu riêng — Tiểu Luận Tốt Nghiệp
        </p>
        <p style='font-size: 13px; margin-bottom: 12px;'>
            Phát triển bởi <b style='color: #333;'>Đặng Anh Kiệt</b> &copy; 2025
        </p>
        <p style='font-size: 12px; color: #999;'>
            <i>Powered by <b>MobileNetV2</b> • <b>Grad-CAM</b> • <b>Google Gemini</b> • <b>Streamlit</b></i>
        </p>
    </div>
    """,
    unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()