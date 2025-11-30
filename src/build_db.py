# src/build_db.py
import os
from dotenv import load_dotenv

# Prefer stable imports across LangChain versions
try:
    from langchain.document_loaders import TextLoader
except Exception:
    # fallback to community loader if present
    from langchain_community.document_loaders import TextLoader

try:
    # langchain v1.x uses plural 'text_splitters'
    from langchain.text_splitters import RecursiveCharacterTextSplitter
except Exception:
    # older versions / variations or separate package
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
    except Exception:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import Chroma
from sentence_transformers import SentenceTransformer

# Use local sentence-transformers embeddings to avoid external API/key issues
class LocalSentenceEmbeddings:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        # returns list[list[float]]
        embs = self.model.encode(texts, show_progress_bar=False)
        # SentenceTransformer returns numpy array
        return [emb.tolist() if hasattr(emb, 'tolist') else list(emb) for emb in embs]

    def embed_query(self, text):
        emb = self.model.encode([text], show_progress_bar=False)[0]
        return emb.tolist() if hasattr(emb, 'tolist') else list(emb)

# Load .env (do NOT hardcode API keys in source)
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    print("⚠️ WARNING: GOOGLE_API_KEY not found in environment. Set it in .env or env vars before running.")

# Đường dẫn (use existing file)
DATA_PATH = "knowledge_base/durian_diseases.txt"
DB_PATH = "knowledge_base/chroma_db"

def create_vector_db():
    print("⏳ Đang tải dữ liệu...")
    # 2. Load dữ liệu từ file text
    loader = TextLoader(DATA_PATH, encoding='utf-8')
    documents = loader.load()

    # 3. Cắt nhỏ văn bản (Chunking)
    # Chunk size 1000 ký tự, overlap 200 để giữ ngữ cảnh giữa các đoạn
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    print(f"✅ Đã chia thành {len(chunks)} đoạn nhỏ.")

    # 4. Tạo Vector DB
    print("⏳ Đang tạo Embeddings (sentence-transformers) và lưu vào ChromaDB...")
    embeddings = LocalSentenceEmbeddings()

    # Tạo và lưu xuống ổ cứng
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_PATH,
    )

    # đảm bảo persist nếu client yêu cầu
    try:
        db.persist()
    except Exception:
        pass

    print("🎉 Thành công! Database đã được lưu tại:", DB_PATH)

if __name__ == "__main__":
    create_vector_db()