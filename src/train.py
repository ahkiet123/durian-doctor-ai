"""
Script huấn luyện mô hình MobileNetV2 cho chẩn đoán bệnh sầu riêng
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import models
from tqdm import tqdm
import time
from datetime import datetime

# Import dataset
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dataset import get_dataloaders, get_class_weights, CLASS_NAMES, DATA_DIR

# Cấu hình
CONFIG = {
    'data_dir': DATA_DIR,
    'model_save_dir': r"c:\Users\ahkie\OneDrive\Desktop\Durian Disease Detection\models",
    'num_classes': 11,
    'batch_size': 32,
    'num_epochs': 30,
    'learning_rate': 1e-4,
    'weight_decay': 1e-4,
    'patience': 5,  # Early stopping patience
    'num_workers': 4,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}


def create_model(num_classes, pretrained=True):
    """
    Tạo model MobileNetV2 với Transfer Learning
    """
    # Load pretrained MobileNetV2
    model = models.mobilenet_v2(weights='IMAGENET1K_V1' if pretrained else None)
    
    # Freeze các layer đầu (feature extractor)
    for param in model.features[:-4].parameters():
        param.requires_grad = False
    
    # Thay đổi classifier layer cuối
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(512, num_classes)
    )
    
    return model


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Train 1 epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc="Training", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        # Forward
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validating", leave=False):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def train(config=CONFIG):
    """Main training function"""
    print("=" * 60)
    print("🚀 BẮT ĐẦU HUẤN LUYỆN MÔ HÌNH")
    print("=" * 60)
    print(f"📱 Device: {config['device']}")
    print(f"📦 Batch size: {config['batch_size']}")
    print(f"🔄 Epochs: {config['num_epochs']}")
    print(f"📈 Learning rate: {config['learning_rate']}")
    print("=" * 60)
    
    device = torch.device(config['device'])
    
    # Load data
    print("\n📂 Loading data...")
    dataloaders = get_dataloaders(
        data_dir=config['data_dir'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers']
    )
    
    # Create model
    print("\n🧠 Creating model...")
    model = create_model(config['num_classes'], pretrained=True)
    model = model.to(device)
    
    # Loss function với class weights
    class_weights = get_class_weights(config['data_dir']).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Optimizer
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Scheduler
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    # Training loop
    best_val_acc = 0.0
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    print("\n" + "=" * 60)
    print("📈 TRAINING PROGRESS")
    print("=" * 60)
    
    start_time = time.time()
    
    for epoch in range(config['num_epochs']):
        print(f"\n🔄 Epoch {epoch+1}/{config['num_epochs']}")
        print("-" * 40)
        
        # Train
        train_loss, train_acc = train_one_epoch(
            model, dataloaders['train'], criterion, optimizer, device
        )
        
        # Validate
        val_loss, val_acc = validate(
            model, dataloaders['val'], criterion, device
        )
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print results
        print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"   Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save model
            os.makedirs(config['model_save_dir'], exist_ok=True)
            save_path = os.path.join(config['model_save_dir'], 'best_mobilenet_v2.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'class_names': CLASS_NAMES,
                'config': config
            }, save_path)
            print(f"   ✅ Saved best model! (Val Acc: {val_acc:.2f}%)")
        else:
            patience_counter += 1
            print(f"   ⏳ No improvement ({patience_counter}/{config['patience']})")
        
        # Early stopping
        if patience_counter >= config['patience']:
            print(f"\n⚠️ Early stopping at epoch {epoch+1}")
            break
    
    # Training completed
    total_time = time.time() - start_time
    
    print("\n" + "=" * 60)
    print("✅ HUẤN LUYỆN HOÀN TẤT!")
    print("=" * 60)
    print(f"⏱️ Thời gian: {total_time/60:.1f} phút")
    print(f"🏆 Best Val Accuracy: {best_val_acc:.2f}%")
    print(f"📉 Best Val Loss: {best_val_loss:.4f}")
    print(f"💾 Model saved: {os.path.join(config['model_save_dir'], 'best_mobilenet_v2.pth')}")
    print("=" * 60)
    
    return model, history


if __name__ == "__main__":
    # Kiểm tra GPU
    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("⚠️ Không tìm thấy GPU, sử dụng CPU")
    
    # Train
    model, history = train()
