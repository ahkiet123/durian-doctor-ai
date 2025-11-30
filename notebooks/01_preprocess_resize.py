"""
Script tiền xử lý ảnh - Resize về 224x224 pixels
Chạy 1 lần để chuẩn bị dữ liệu cho training
"""

import os
from PIL import Image
from tqdm import tqdm
import shutil

# Paths
SOURCE_BASE = r"c:\Users\ahkie\OneDrive\Desktop\Durian Disease Detection\data\mendeley_dataset\dataset"
DEST_BASE = r"c:\Users\ahkie\OneDrive\Desktop\Durian Disease Detection\data\processed_train_224"

# Target size
TARGET_SIZE = (224, 224)

def resize_and_save(src_path, dst_path):
    """Resize ảnh về 224x224 và lưu"""
    try:
        img = Image.open(src_path).convert('RGB')
        img_resized = img.resize(TARGET_SIZE, Image.LANCZOS)
        img_resized.save(dst_path, quality=95)
        return True
    except Exception as e:
        print(f"Lỗi xử lý {src_path}: {e}")
        return False

def process_folder(folder_name):
    """Xử lý một folder (Train, Validation, Test)"""
    src_folder = os.path.join(SOURCE_BASE, folder_name)
    dst_folder = os.path.join(DEST_BASE, folder_name)
    
    if not os.path.exists(src_folder):
        print(f"Không tìm thấy folder: {src_folder}")
        return
    
    # Lấy danh sách các class
    classes = [d for d in os.listdir(src_folder) 
               if os.path.isdir(os.path.join(src_folder, d))]
    
    total_processed = 0
    total_errors = 0
    
    for class_name in classes:
        src_class_path = os.path.join(src_folder, class_name)
        dst_class_path = os.path.join(dst_folder, class_name)
        
        # Tạo folder đích
        os.makedirs(dst_class_path, exist_ok=True)
        
        # Lấy danh sách ảnh
        images = [f for f in os.listdir(src_class_path) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        print(f"\n📁 {folder_name}/{class_name}: {len(images)} ảnh")
        
        for img_name in tqdm(images, desc=f"  Processing", leave=False):
            src_path = os.path.join(src_class_path, img_name)
            
            # Đổi extension sang .jpg để thống nhất
            new_name = os.path.splitext(img_name)[0] + '.jpg'
            dst_path = os.path.join(dst_class_path, new_name)
            
            if resize_and_save(src_path, dst_path):
                total_processed += 1
            else:
                total_errors += 1
    
    return total_processed, total_errors

def main():
    print("=" * 60)
    print("🔄 TIỀN XỬ LÝ ẢNH - RESIZE VỀ 224x224")
    print("=" * 60)
    print(f"📂 Nguồn: {SOURCE_BASE}")
    print(f"📂 Đích: {DEST_BASE}")
    print(f"📐 Kích thước: {TARGET_SIZE}")
    print("=" * 60)
    
    # Tạo folder đích
    os.makedirs(DEST_BASE, exist_ok=True)
    
    total_all = 0
    errors_all = 0
    
    # Xử lý từng folder
    for folder in ['Train', 'Validation', 'Test']:
        print(f"\n{'='*40}")
        print(f"📦 Đang xử lý: {folder}")
        print('='*40)
        
        processed, errors = process_folder(folder)
        total_all += processed
        errors_all += errors
    
    # Tổng kết
    print("\n" + "=" * 60)
    print("✅ HOÀN TẤT!")
    print(f"   - Tổng ảnh đã xử lý: {total_all}")
    print(f"   - Lỗi: {errors_all}")
    print(f"   - Đã lưu tại: {DEST_BASE}")
    print("=" * 60)

if __name__ == "__main__":
    main()
