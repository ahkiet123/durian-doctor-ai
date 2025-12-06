"""
Classifier Module - Load model và dự đoán bệnh sầu riêng
"""
import os
import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import streamlit as st

# --- CẤU HÌNH CLASS BỆNH (11 Lớp) ---
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
        
        # Đường dẫn model
        model_path = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'best_mobilenet_v2.pth')
        
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
        print(f"Error loading model: {e}")
        return None, False


def predict_and_gradcam(image, model):
    """Dự đoán và tạo Grad-CAM heatmap"""
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
