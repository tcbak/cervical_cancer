import torch
import torch.nn as nn
import torchvision.models as models

def get_model(num_classes=4, model_name="efficientnet_b0", pretrained=True, freeze_backbone=False):
    """
    Builds and returns a PyTorch image classification model.
    Supported backbones:
      - 'efficientnet_b0'
      - 'resnet18'
      - 'resnet50'
    """
    print(f"Initializing {model_name} (pretrained={pretrained}, freeze_backbone={freeze_backbone})...")
    
    if model_name == "efficientnet_b0":
        weights = "DEFAULT" if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
        else:
            for param in model.parameters():
                param.requires_grad = True
                
        # Replace classification head
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, num_classes)
        )
        
    elif model_name == "resnet18":
        weights = "DEFAULT" if pretrained else None
        model = models.resnet18(weights=weights)
        
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
        else:
            for param in model.parameters():
                param.requires_grad = True
                
        # Replace FC layer
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        
    elif model_name == "resnet50":
        weights = "DEFAULT" if pretrained else None
        model = models.resnet50(weights=weights)
        
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
        else:
            for param in model.parameters():
                param.requires_grad = True
                
        # Replace FC layer
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")
        
    return model

if __name__ == "__main__":
    print("Testing Model Module...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(num_classes=4, model_name="efficientnet_b0")
    model = model.to(device)
    
    # Test pass
    dummy_input = torch.randn(2, 3, 224, 224).to(device)
    output = model(dummy_input)
    print("Dummy input shape: [2, 3, 224, 224]")
    print(f"Model output shape: {output.shape} (Expected: [2, 4])")
    print("Model loaded successfully!")
