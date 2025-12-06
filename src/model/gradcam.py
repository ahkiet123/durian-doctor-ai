"""
Grad-CAM Module - Explainable AI với heatmap visualization
Chứa GradCAMExplainer class cho việc giải thích chi tiết
"""
import torch
import numpy as np
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import torchvision.transforms as transforms


class GradCAMExplainer:
    """
    Class để tạo heatmap giải thích cho model MobileNetV2
    Sử dụng khi cần tuỳ chỉnh chi tiết hơn so với predict_and_gradcam()
    """
    
    def __init__(self, model, device='cpu'):
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
        Tạo heatmap cho một ảnh từ đường dẫn file
        
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
            targets = None
        else:
            targets = [ClassifierOutputTarget(target_class)]
        
        grayscale_cam = self.cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        
        # Tạo heatmap overlay
        heatmap_overlay = show_cam_on_image(
            original_image.astype(np.float32), 
            grayscale_cam, 
            use_rgb=True
        )
        
        return original_image, heatmap_overlay, predicted_class, confidence


def visualize_prediction(image_path, model, class_names, save_path=None):
    """
    Hàm tiện ích để visualize kết quả với matplotlib
    Dùng cho notebooks hoặc standalone scripts
    """
    import matplotlib.pyplot as plt
    
    explainer = GradCAMExplainer(model, device='cpu')
    original, heatmap, pred_class, conf = explainer.generate_heatmap(image_path)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].imshow(original)
    axes[0].set_title('Ảnh gốc', fontsize=14)
    axes[0].axis('off')
    
    axes[1].imshow(heatmap)
    axes[1].set_title(f'Grad-CAM: {class_names[pred_class]}\nĐộ tin cậy: {conf*100:.1f}%', fontsize=14)
    axes[1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Đã lưu kết quả tại: {save_path}")
    
    plt.show()
    return pred_class, conf
