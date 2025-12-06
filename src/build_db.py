"""
Build Chroma Vector Database từ Knowledge Base
Chạy một lần để tạo DB, sau đó app sẽ tự load.
"""
import os
from rag_engine import build_chroma_db_if_missing

# Đường dẫn
BASE_DIR = os.path.dirname(__file__)
KB_PATH = os.path.join(BASE_DIR, '..', 'knowledge_base', 'durian_document.txt')
DB_PATH = os.path.join(BASE_DIR, '..', 'knowledge_base', 'chroma_db')

def main():
    print("=" * 50)
    print("🌳 Durian Doctor - Build Vector Database")
    print("=" * 50)
    
    if os.path.exists(DB_PATH) and os.listdir(DB_PATH):
        print(f"⚠️ Database đã tồn tại tại: {DB_PATH}")
        response = input("Bạn có muốn xóa và tạo lại? (y/n): ")
        if response.lower() != 'y':
            print("❌ Đã hủy.")
            return
        import shutil
        shutil.rmtree(DB_PATH)
        print("🗑️ Đã xóa database cũ.")
    
    db = build_chroma_db_if_missing(KB_PATH, DB_PATH)
    
    if db:
        print("✅ Hoàn tất!")
    else:
        print("❌ Có lỗi xảy ra.")

if __name__ == "__main__":
    main()