"""
Script đánh giá mô hình trên tập Test
"""

import os
import sys
import torch
import torch.nn as nn
from torchvision import models
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm

# Import dataset
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dataset import get_dataloaders, CLASS_NAMES, CLASS_NAMES_VI, DATA_DIR

# Paths
MODEL_PATH = r"c:\Users\ahkie\OneDrive\Desktop\Durian Disease Detection\models\best_mobilenet_v2.pth"
RESULTS_DIR = r"c:\Users\ahkie\OneDrive\Desktop\Durian Disease Detection\results"


def load_model(model_path, num_classes=11, device='cuda'):
    """Load trained model"""
    # Tạo model architecture
    model = models.mobilenet_v2(weights=None)
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(512, num_classes)
    )
    
    # Load weights
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"✅ Loaded model from epoch {checkpoint['epoch']}")
    print(f"   Val Accuracy: {checkpoint['val_acc']:.2f}%")
    
    return model, checkpoint


def evaluate(model, dataloader, device):
    """Evaluate model and get predictions"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            outputs = model(images)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = outputs.max(1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(14, 12))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title('Confusion Matrix', fontsize=16)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"💾 Saved confusion matrix: {save_path}")
    
    plt.show()


def plot_classification_metrics(report_dict, save_path=None):
    """Plot precision, recall, f1-score per class"""
    classes = list(report_dict.keys())[:-3]  # Exclude accuracy, macro avg, weighted avg
    
    precision = [report_dict[c]['precision'] for c in classes]
    recall = [report_dict[c]['recall'] for c in classes]
    f1 = [report_dict[c]['f1-score'] for c in classes]
    
    x = np.arange(len(classes))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(14, 6))
    bars1 = ax.bar(x - width, precision, width, label='Precision', color='#2ecc71')
    bars2 = ax.bar(x, recall, width, label='Recall', color='#3498db')
    bars3 = ax.bar(x + width, f1, width, label='F1-Score', color='#e74c3c')
    
    ax.set_xlabel('Classes')
    ax.set_ylabel('Score')
    ax.set_title('Classification Metrics per Class')
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"💾 Saved metrics plot: {save_path}")
    
    plt.show()


def main():
    print("=" * 60)
    print("📊 ĐÁNH GIÁ MÔ HÌNH TRÊN TẬP TEST")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📱 Device: {device}")
    
    # Check model exists
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Không tìm thấy model: {MODEL_PATH}")
        print("   Vui lòng chạy train.py trước!")
        return
    
    # Load model
    print("\n🧠 Loading model...")
    model, checkpoint = load_model(MODEL_PATH, device=device)
    
    # Load test data
    print("\n📂 Loading test data...")
    dataloaders = get_dataloaders(batch_size=32, num_workers=0)
    
    # Evaluate
    print("\n🔍 Evaluating...")
    preds, labels, probs = evaluate(model, dataloaders['test'], device)
    
    # Calculate accuracy
    accuracy = (preds == labels).mean() * 100
    print(f"\n🎯 Test Accuracy: {accuracy:.2f}%")
    
    # Classification report
    print("\n" + "=" * 60)
    print("📋 CLASSIFICATION REPORT")
    print("=" * 60)
    
    # Tạo short names cho report
    short_names = [name[:15] for name in CLASS_NAMES]
    report = classification_report(labels, preds, target_names=short_names)
    print(report)
    
    # Get report as dict for plotting
    report_dict = classification_report(labels, preds, target_names=short_names, output_dict=True)
    
    # Create results directory
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Save report to file
    report_path = os.path.join(RESULTS_DIR, 'classification_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"Test Accuracy: {accuracy:.2f}%\n\n")
        f.write(report)
    print(f"💾 Saved report: {report_path}")
    
    # Plot confusion matrix
    print("\n📊 Generating confusion matrix...")
    cm_path = os.path.join(RESULTS_DIR, 'confusion_matrix.png')
    plot_confusion_matrix(labels, preds, short_names, cm_path)
    
    # Plot metrics
    print("\n📈 Generating metrics plot...")
    metrics_path = os.path.join(RESULTS_DIR, 'classification_metrics.png')
    plot_classification_metrics(report_dict, metrics_path)
    
    print("\n" + "=" * 60)
    print("✅ ĐÁNH GIÁ HOÀN TẤT!")
    print("=" * 60)
    print(f"📁 Kết quả lưu tại: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
