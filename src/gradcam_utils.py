"""
Grad-CAM Utilities for Explainable AI (XAI)
Tạo Heatmap để giải thích quyết định của mô hình
"""

import torch
import numpy as np
import cv2
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import torchvision.transforms as transforms


class GradCAMExplainer:
    """
    Class để tạo heatmap giải thích cho model MobileNetV2
    """
    
    def __init__(self, model, device='cuda'):
        """
        Args:
            model: Model PyTorch đã train (MobileNetV2)
            device: 'cuda' hoặc 'cpu'
        """
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
        
        # Lấy layer cuối cùng của MobileNetV2 để tạo CAM
        # MobileNetV2 có cấu trúc: features -> classifier
        self.target_layers = [model.features[-1]]
        
        # Khởi tạo GradCAM
        self.cam = GradCAM(model=self.model, target_layers=self.target_layers)
        
        # Transform cho ảnh đầu vào
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def generate_heatmap(self, image_path, target_class=None):
        """
        Tạo heatmap cho một ảnh
        
        Args:
            image_path: Đường dẫn tới ảnh
            target_class: Class muốn giải thích (None = class được dự đoán)
            
        Returns:
            - original_image: Ảnh gốc (numpy array)
            - heatmap_overlay: Ảnh với heatmap đè lên
            - predicted_class: Class được dự đoán
            - confidence: Độ tin cậy
        """
        # Đọc và xử lý ảnh
        pil_image = Image.open(image_path).convert('RGB')
        
        # Resize ảnh gốc về 224x224 để hiển thị
        pil_image_resized = pil_image.resize((224, 224))
        original_image = np.array(pil_image_resized) / 255.0
        
        # Transform cho model
        input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        
        # Dự đoán class
        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            predicted_class = output.argmax(dim=1).item()
            confidence = probabilities[0, predicted_class].item()
        
        # Tạo CAM
        if target_class is None:
            targets = None  # Sử dụng class được dự đoán
        else:
            targets = [ClassifierOutputTarget(target_class)]
        
        # Tạo grayscale CAM
        grayscale_cam = self.cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]  # Lấy CAM của ảnh đầu tiên
        
        # Tạo heatmap overlay
        heatmap_overlay = show_cam_on_image(
            original_image.astype(np.float32), 
            grayscale_cam, 
            use_rgb=True
        )
        
        return original_image, heatmap_overlay, predicted_class, confidence
    
    def generate_heatmap_from_tensor(self, image_tensor, original_image_np, target_class=None):
        """
        Tạo heatmap từ tensor đã được transform
        
        Args:
            image_tensor: Tensor ảnh đã transform (1, 3, 224, 224)
            original_image_np: Ảnh gốc dạng numpy (224, 224, 3), giá trị 0-1
            target_class: Class muốn giải thích
            
        Returns:
            - heatmap_overlay: Ảnh với heatmap đè lên
            - predicted_class: Class được dự đoán
            - confidence: Độ tin cậy
        """
        input_tensor = image_tensor.to(self.device)
        
        # Dự đoán
        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            predicted_class = output.argmax(dim=1).item()
            confidence = probabilities[0, predicted_class].item()
        
        # Tạo CAM
        if target_class is None:
            targets = None
        else:
            targets = [ClassifierOutputTarget(target_class)]
        
        grayscale_cam = self.cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        
        # Tạo heatmap overlay
        heatmap_overlay = show_cam_on_image(
            original_image_np.astype(np.float32), 
            grayscale_cam, 
            use_rgb=True
        )
        
        return heatmap_overlay, predicted_class, confidence


def visualize_prediction(image_path, model, class_names, device='cuda', save_path=None):
    """
    Hàm tiện ích để visualize kết quả dự đoán với heatmap
    
    Args:
        image_path: Đường dẫn ảnh
        model: Model đã train
        class_names: List tên các class
        device: 'cuda' hoặc 'cpu'
        save_path: Đường dẫn lưu ảnh kết quả (optional)
    """
    import matplotlib.pyplot as plt
    
    explainer = GradCAMExplainer(model, device)
    original, heatmap, pred_class, conf = explainer.generate_heatmap(image_path)
    
    # Tạo figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Ảnh gốc
    axes[0].imshow(original)
    axes[0].set_title('Ảnh gốc', fontsize=14)
    axes[0].axis('off')
    
    # Ảnh với heatmap
    axes[1].imshow(heatmap)
    axes[1].set_title(f'Grad-CAM: {class_names[pred_class]}\nĐộ tin cậy: {conf*100:.1f}%', fontsize=14)
    axes[1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Đã lưu kết quả tại: {save_path}")
    
    plt.show()
    
    return pred_class, conf


if __name__ == "__main__":
    # Test code
    print("Grad-CAM Utils đã sẵn sàng!")
    print("Sử dụng: from gradcam_utils import GradCAMExplainer, visualize_prediction")
