"""
Dataset class cho PyTorch - Load dữ liệu bệnh sầu riêng
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


# Danh sách 11 classes
CLASS_NAMES = [
    'anthracnose_disease',
    'canker_disease', 
    'fruit_rot',
    'leaf_healthy',
    'mealybug_infestation',
    'pink_disease',
    'sooty_mold',
    'stem_blight',
    'stem_cracking_ gummosis',
    'thrips_disease',
    'yellow_leaf'
]

# Tên tiếng Việt cho hiển thị
CLASS_NAMES_VI = {
    'anthracnose_disease': 'Bệnh thán thư (Anthracnose)',
    'canker_disease': 'Bệnh loét (Canker)',
    'fruit_rot': 'Thối trái (Fruit Rot)',
    'leaf_healthy': 'Lá khỏe mạnh (Healthy)',
    'mealybug_infestation': 'Rệp sáp (Mealybug)',
    'pink_disease': 'Bệnh hồng (Pink Disease)',
    'sooty_mold': 'Nấm muội đen (Sooty Mold)',
    'stem_blight': 'Cháy thân (Stem Blight)',
    'stem_cracking_ gummosis': 'Nứt thân xì mủ (Gummosis)',
    'thrips_disease': 'Bọ trĩ (Thrips)',
    'yellow_leaf': 'Vàng lá (Yellow Leaf)'
}

# Đường dẫn mặc định
DATA_DIR = r"c:\Users\ahkie\OneDrive\Desktop\Durian Disease Detection\data\processed_train_224"


class DurianDiseaseDataset(Dataset):
    """
    Custom Dataset cho bệnh sầu riêng
    """
    
    def __init__(self, data_dir, split='Train', transform=None):
        """
        Args:
            data_dir: Đường dẫn tới folder chứa Train/Validation/Test
            split: 'Train', 'Validation', hoặc 'Test'
            transform: Các phép biến đổi ảnh
        """
        self.data_dir = os.path.join(data_dir, split)
        self.split = split
        self.transform = transform
        
        # Load danh sách ảnh và nhãn
        self.samples = []
        self.class_to_idx = {cls: idx for idx, cls in enumerate(CLASS_NAMES)}
        
        for class_name in CLASS_NAMES:
            class_path = os.path.join(self.data_dir, class_name)
            if os.path.exists(class_path):
                for img_name in os.listdir(class_path):
                    if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(class_path, img_name)
                        self.samples.append((img_path, self.class_to_idx[class_name]))
        
        print(f"📦 Loaded {split}: {len(self.samples)} samples, {len(CLASS_NAMES)} classes")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load ảnh
        image = Image.open(img_path).convert('RGB')
        
        # Apply transform
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_transforms(split='Train'):
    """
    Trả về transforms phù hợp cho từng split
    
    Args:
        split: 'Train', 'Validation', hoặc 'Test'
    """
    # Normalize theo ImageNet (vì dùng pretrained MobileNetV2)
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    if split == 'Train':
        # Data Augmentation cho training
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            normalize
        ])
    else:
        # Không augmentation cho validation/test
        return transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])


def get_dataloaders(data_dir=DATA_DIR, batch_size=32, num_workers=4):
    """
    Tạo DataLoader cho Train, Validation, Test
    
    Args:
        data_dir: Đường dẫn data
        batch_size: Batch size
        num_workers: Số worker load data
        
    Returns:
        dict với keys: 'train', 'val', 'test'
    """
    dataloaders = {}
    
    # Train
    train_dataset = DurianDiseaseDataset(
        data_dir, 
        split='Train', 
        transform=get_transforms('Train')
    )
    dataloaders['train'] = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Validation
    val_dataset = DurianDiseaseDataset(
        data_dir, 
        split='Validation', 
        transform=get_transforms('Validation')
    )
    dataloaders['val'] = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Test
    test_dataset = DurianDiseaseDataset(
        data_dir, 
        split='Test', 
        transform=get_transforms('Test')
    )
    dataloaders['test'] = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloaders


def get_class_weights(data_dir=DATA_DIR):
    """
    Tính class weights để xử lý imbalanced data
    """
    train_dir = os.path.join(data_dir, 'Train')
    class_counts = []
    
    for class_name in CLASS_NAMES:
        class_path = os.path.join(train_dir, class_name)
        if os.path.exists(class_path):
            count = len([f for f in os.listdir(class_path) if f.endswith(('.jpg', '.png'))])
            class_counts.append(count)
        else:
            class_counts.append(0)
    
    # Inverse frequency weighting
    total = sum(class_counts)
    weights = [total / (len(class_counts) * c) if c > 0 else 0 for c in class_counts]
    
    return torch.FloatTensor(weights)


if __name__ == "__main__":
    # Test dataset
    print("=" * 50)
    print("🧪 TEST DATASET")
    print("=" * 50)
    
    # Load dataloaders
    dataloaders = get_dataloaders(batch_size=32, num_workers=0)
    
    # Thống kê
    print(f"\n📊 Thống kê:")
    print(f"   - Train batches: {len(dataloaders['train'])}")
    print(f"   - Val batches: {len(dataloaders['val'])}")
    print(f"   - Test batches: {len(dataloaders['test'])}")
    
    # Test load 1 batch
    images, labels = next(iter(dataloaders['train']))
    print(f"\n🖼️ Sample batch:")
    print(f"   - Images shape: {images.shape}")
    print(f"   - Labels shape: {labels.shape}")
    
    # Class weights
    weights = get_class_weights()
    print(f"\n⚖️ Class weights:")
    for i, (name, w) in enumerate(zip(CLASS_NAMES, weights)):
        print(f"   {i}: {name[:20]:20s} - {w:.3f}")
