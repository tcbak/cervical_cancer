import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.forward_hook = self.target_layer.register_forward_hook(self.save_activation)
        self.backward_hook = self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def remove_hooks(self):
        """Removes the hooks to prevent memory leaks."""
        self.forward_hook.remove()
        self.backward_hook.remove()

    def generate(self, input_tensor, class_idx):
        self.model.zero_grad()
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = torch.argmax(output)
            
        loss = output[0, class_idx]
        loss.backward()
        
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations.clone()
        
        for i in range(activations.size(1)):
            activations[:, i, :, :] *= pooled_gradients[i]
            
        heatmap = torch.mean(activations, dim=1).squeeze()
        heatmap = F.relu(heatmap)
        heatmap /= (torch.max(heatmap) + 1e-8)
        
        return heatmap.detach().cpu().numpy(), output

def get_target_layer(model, model_name="efficientnet_b0"):
    """
    Returns the target layer for Grad-CAM based on model architecture.
    """
    if "efficientnet" in model_name.lower():
        # Typically model.features[8] is the last block of EfficientNet-B0
        if hasattr(model, "features") and len(model.features) > 8:
            return model.features[8]
        else:
            return model.features[-1]
    elif "resnet" in model_name.lower():
        # resnet uses layer4
        return model.layer4
    else:
        # Fallback to last child module that is a Conv2d or container
        conv_modules = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
        if len(conv_modules) > 0:
            return conv_modules[-1]
        raise ValueError("Cannot determine target layer for Grad-CAM automatically.")

def localize_lesion(heatmap, image_shape, threshold=0.4):
    """
    Uses thresholding and contour detection on the Grad-CAM heatmap 
    to localize the coordinates of the suspected lesion.
    Returns:
      - bbox: (x, y, w, h) of the primary detected region
      - contours: OpenCV contours
      - mask: Binary threshold mask
    """
    # Resize heatmap to match original image size
    heatmap_resized = cv2.resize(heatmap, (image_shape[1], image_shape[0]))
    
    # Threshold heatmap to create a binary mask of high activation areas
    _, mask = cv2.threshold((heatmap_resized * 255).astype(np.uint8), int(threshold * 255), 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    bbox = None
    if contours:
        # Find the contour with the largest area
        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) > 50:  # Ignore tiny noise
            bbox = cv2.boundingRect(largest_contour)
            
    return bbox, contours, mask

def create_explainability_overlay(image_path, model, transform, model_name="efficientnet_b0", device="cpu", threshold=0.4):
    """
    Generates Grad-CAM and overlays contours & bounding box on the original image.
    """
    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    img_np = np.array(image)
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Find target layer
    target_layer = get_target_layer(model, model_name)
    
    # Initialize GradCAM
    cam = GradCAM(model, target_layer)
    
    try:
        # Run model and generate heatmap
        heatmap, outputs = cam.generate(input_tensor, class_idx=None)
    finally:
        cam.remove_hooks()
        
    probs = F.softmax(outputs, dim=1)[0]
    conf, pred_idx = torch.max(probs, 0)
    
    # Perform localization
    bbox, contours, mask = localize_lesion(heatmap, img_np.shape, threshold=threshold)
    
    # Build visual overlays
    heatmap_resized = cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_color_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    
    # Composite overlay
    overlay = cv2.addWeighted(img_np, 0.6, heatmap_color_rgb, 0.4, 0)
    
    # Draw outline and bounding box on overlay copy
    annotated_overlay = overlay.copy()
    if contours:
        # Draw all contours in green
        cv2.drawContours(annotated_overlay, contours, -1, (0, 255, 0), 2)
    if bbox:
        # Draw bounding box in red
        x, y, w, h = bbox
        cv2.rectangle(annotated_overlay, (x, y), (x + w, y + h), (255, 0, 0), 3)
        
    return {
        "original": img_np,
        "heatmap": heatmap_resized,
        "overlay": overlay,
        "annotated": annotated_overlay,
        "pred_idx": pred_idx.item(),
        "confidence": conf.item(),
        "bbox": bbox
    }

if __name__ == "__main__":
    print("Testing Explainability Module...")
    # This block can be run if an image exists
    # Requires a mock model and image to run cleanly, which is tested in the pipeline run.
    pass
