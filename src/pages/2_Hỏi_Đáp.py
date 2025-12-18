"""
Page 2: Hỏi đáp AI (Chatbot RAG)
Giao diện chat fullscreen giống các AI chat hiện đại
"""
import streamlit as st
import sys
import os

# Thêm path để import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_engine import load_vector_db
from prompts.system_prompt import SYSTEM_PROMPT, build_full_prompt
from config import setup_gemini, get_gemini_client, get_gemini_model_name

# Setup
GROQ_API_KEY = setup_gemini()  # Tên hàm giữ nguyên để tương thích
groq_client = get_gemini_client()  # Tên giữ nguyên nhưng trả về Groq client



# Load vector DB
vector_db = load_vector_db()

# Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# === HEADER CỐ ĐỊNH ===
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown("### 💬 Hỏi đáp với Chuyên gia AI")
with col2:
    show_thinking = st.toggle("🧠 Suy nghĩ", value=False, help="Xem AI đang làm gì")

# Hiển thị kết quả chẩn đoán gần nhất (nếu có)
if 'diagnosis_vi' in st.session_state:
    st.info(f"📋 Chẩn đoán gần nhất: **{st.session_state['diagnosis_vi']}**")

st.markdown("---")

# === HIỂN THỊ TẤT CẢ MESSAGES (scroll tự nhiên theo trang) ===
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# === INPUT Ở DƯỚI CÙNG ===
prompt = st.chat_input("Hỏi về bệnh sầu riêng, cách điều trị...")

if prompt:
    # Thêm và hiển thị message user
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Xử lý response
    with st.chat_message("assistant"):
        if not GROQ_API_KEY:
            st.warning("⚠️ Vui lòng cấu hình GROQ_API_KEY trong file .env")
        else:
            thinking_container = st.empty()
            
            # === STEP 1: Tìm kiếm RAG ===
            if show_thinking:
                with thinking_container.container():
                    st.markdown("🔍 **Đang tìm kiếm trong cơ sở tri thức...**")
                    with st.status("Truy vấn RAG Database", expanded=True):
                        st.write("📚 Kết nối ChromaDB...")
            
            retrieved_block = ""
            retrieved_docs_display = []
            has_relevant_docs = False
            
            # Các từ khóa liên quan để xác định cần RAG
            durian_keywords = ['sầu riêng', 'durian', 'bệnh', 'lá', 'trái', 'thân', 'rễ', 'thuốc', 'phân', 'bón', 
                               'phun', 'trị', 'chữa', 'triệu chứng', 'vàng', 'thối', 'nấm', 'sâu', 
                               'côn trùng', 'rệp', 'nhện', 'xì mủ', 'nứt', 'cháy', 'héo', 'chăm sóc',
                               'tưới', 'cắt tỉa', 'ra hoa', 'đậu trái', 'thu hoạch', 'giống', 'ri6', 'monthong', 'thái',
                               'cây', 'vườn', 'nhà vườn', 'nông dân', 'thương lái', 'musang', 'dona']
            
            query_lower = prompt.lower()
            is_durian_related = any(kw in query_lower for kw in durian_keywords)
            
            try:
                if vector_db and is_durian_related:
                    docs = vector_db.similarity_search(prompt, k=3)
                    if docs:
                        for i, d in enumerate(docs):
                            if len(d.page_content) > 50:
                                retrieved_docs_display.append(f"**[{i+1}]** {d.page_content[:150]}...")
                                has_relevant_docs = True
                        if has_relevant_docs:
                            content_list = [f"[{i+1}] {d.page_content}" for i, d in enumerate(docs)]
                            retrieved_block = "THÔNG TIN THAM KHẢO TỪ TÀI LIỆU (RAG):\n" + "\n\n".join(content_list)
            except Exception as e:
                print(f"RAG Error: {e}")
            
            if show_thinking:
                with thinking_container.container():
                    with st.status("Truy vấn RAG Database", expanded=True, state="complete"):
                        if has_relevant_docs:
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
                    with st.status("Xây dựng ngữ cảnh", expanded=True):
                        st.write("📝 Phân tích lịch sử hội thoại...")
            
            diag_context = ""
            if 'diagnosis_vi' in st.session_state:
                diag_context = f"LƯU Ý NGỮ CẢNH: Người dùng vừa upload ảnh và được AI chẩn đoán cây bị bệnh: {st.session_state['diagnosis_vi']}."

            chat_history_text = ""
            recent_msgs = st.session_state.messages[-6:]
            for msg in recent_msgs:
                role_label = "Người dùng" if msg["role"] == "user" else "Durian Doctor"
                chat_history_text += f"{role_label}: {msg['content']}\n"

            full_prompt = build_full_prompt(
                SYSTEM_PROMPT, retrieved_block, diag_context, 
                chat_history_text, prompt
            )
            
            # === STEP 3: Gọi Groq LLM ===
            if show_thinking:
                with thinking_container.container():
                    with st.status("Truy vấn RAG Database", expanded=False, state="complete"):
                        st.write("✅ Hoàn tất")
                    with st.status("Xây dựng ngữ cảnh", expanded=False, state="complete"):
                        st.write("✅ Hoàn tất")
                    with st.status("🤖 Groq AI đang suy nghĩ...", expanded=True):
                        st.write("💭 Phân tích câu hỏi và tài liệu...")
            
            try:
                # Groq API: Sử dụng OpenAI-compatible chat completions
                model_name = get_gemini_model_name()
                response = groq_client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "user", "content": full_prompt}
                    ],
                    temperature=0.7,
                    max_tokens=2048
                )
                bot_reply = response.choices[0].message.content
            except Exception as e:
                bot_reply = f"⚠️ Lỗi kết nối Groq API: {e}"


            
            # === Hoàn tất ===
            if show_thinking:
                with thinking_container.container():
                    with st.status("Truy vấn RAG Database", expanded=False, state="complete"):
                        st.write("✅ Hoàn tất")
                    with st.status("Xây dựng ngữ cảnh", expanded=False, state="complete"):
                        st.write("✅ Hoàn tất")
                    with st.status("🤖 Groq AI đang suy nghĩ...", expanded=False, state="complete"):
                        st.write("✅ Đã tạo câu trả lời")
                    st.markdown("---")
            else:
                thinking_container.empty()
            
            # Hiển thị response
            st.markdown(bot_reply)
            
            # Chỉ hiển thị nguồn tài liệu khi có docs liên quan
            if has_relevant_docs and retrieved_docs_display:
                with st.expander("📚 Nguồn tài liệu tham khảo", expanded=False):
                    for i, doc in enumerate(retrieved_docs_display):
                        st.markdown(doc)
                        if i < len(retrieved_docs_display) - 1:
                            st.divider()
            
            # Lưu message
            st.session_state.messages.append({"role": "assistant", "content": bot_reply})
