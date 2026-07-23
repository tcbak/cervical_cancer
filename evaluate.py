import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix

from dataset import get_dataloaders
from model import get_model

def evaluate_model(args):
    # Setup directories
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Dataloaders
    print("Loading test dataset...")
    _, _, test_loader, class_names, class_weights = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size
    )
    num_classes = len(class_names)
    print(f"Dataset classes: {class_names}")
    
    # 2. Load Model
    model = get_model(
        num_classes=num_classes,
        model_name=args.model_name,
        pretrained=False
    )
    
    print(f"Loading checkpoint weights from {args.model_path}...")
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {args.model_path}")
        
    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    # Loss function for evaluation
    loss_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=loss_weights)
    
    # 3. Running evaluation
    test_loss = 0.0
    correct = 0
    total = 0
    
    all_preds = []
    all_targets = []
    
    print("\nRunning test evaluation...")
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            
    avg_test_loss = test_loss / total
    test_acc = (correct / total) * 100
    
    print("\n================ TEST SET METRICS =================\n")
    print(f"Average Test Loss: {avg_test_loss:.4f}")
    print(f"Overall Accuracy:  {test_acc:.2f}%")
    print("\nClassification Report:")
    report = classification_report(all_targets, all_preds, target_names=class_names)
    print(report)
    
    # Save classification report text
    report_path = os.path.join(args.output_dir, "classification_report.txt")
    with open(report_path, 'w') as f:
        f.write(f"Average Test Loss: {avg_test_loss:.4f}\n")
        f.write(f"Overall Accuracy:  {test_acc:.2f}%\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    print(f"Saved text report to {report_path}")
    
    # 4. Generate & Save Confusion Matrix
    cm = confusion_matrix(all_targets, all_preds)
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(8, 6))
    plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Normalized Confusion Matrix')
    plt.colorbar()
    
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Loop to annotate matrix entries
    fmt = '.2f'
    thresh = cm_normalized.max() / 2.
    for i in range(cm_normalized.shape[0]):
        for j in range(cm_normalized.shape[1]):
            plt.text(j, i, format(cm_normalized[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm_normalized[i, j] > thresh else "black")
            
    plt.tight_layout()
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    
    cm_plot_path = os.path.join(args.output_dir, 'confusion_matrix.png')
    plt.savefig(cm_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved normalized confusion matrix plot to {cm_plot_path}")
    print("===================================================\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Cervical Cancer Model")
    parser.add_argument("--data_dir", type=str, default="D:/MSC AI/Split_Data", help="Path to Split_Data directory")
    parser.add_argument("--model_path", type=str, default="D:/MSC AI/cervical_classifier/outputs/best_cervical_model.pth", help="Path to trained .pth weights")
    parser.add_argument("--model_name", type=str, default="efficientnet_b0", help="Model backbone (efficientnet_b0, resnet18, resnet50)")
    parser.add_argument("--output_dir", type=str, default="D:/MSC%20AI/cervical_classifier/outputs", help="Directory to save eval reports and figures")
    parser.add_argument("--batch_size", type=int, default=32, help="DataLoader batch size")
    
    args = parser.parse_args()
    # Normalize path formatting for local compatibility
    args.model_path = args.model_path.replace("%20", " ")
    args.output_dir = args.output_dir.replace("%20", " ")
    evaluate_model(args)
