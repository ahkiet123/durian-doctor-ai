import os
import streamlit as st
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- CẤU HÌNH RAG (LOCAL EMBEDDINGS) ---
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
        # Xác định đường dẫn base dựa trên vị trí file hiện tại (src/rag_engine.py)
        base_dir = os.path.dirname(__file__)
        kb_path = os.path.join(base_dir, '..', 'knowledge_base', 'durian_diseases.txt')
        
        # Thử local path trước
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
