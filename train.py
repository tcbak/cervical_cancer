import os
import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, classification_report

from dataset import get_dataloaders
from model import get_model

def train_model(args):
    # Setup directories
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Dataloaders
    print("Loading datasets...")
    train_loader, val_loader, _, class_names, class_weights = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size
    )
    num_classes = len(class_names)
    print(f"Dataset classes: {class_names}")
    print(f"Computed loss weights for imbalance handling: {class_weights}")
    
    # 2. Model
    model = get_model(
        num_classes=num_classes,
        model_name=args.model_name,
        pretrained=not args.no_pretrained,
        freeze_backbone=args.freeze_backbone
    )
    model = model.to(device)
    
    # 3. Loss & Optimizer
    # Note: WeightedRandomSampler already balances batch distributions, but adding 
    # class weights to the loss function provides an extra layer of protection.
    loss_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=loss_weights)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    
    # History tracking
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": []
    }
    
    best_val_f1 = 0.0
    patience_counter = 0
    
    print("\nStarting Training Loop...")
    for epoch in range(args.epochs):
        # --- Training Phase ---
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
            # Print batch info occasionally
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(train_loader):
                print(f"Epoch [{epoch+1}/{args.epochs}] | Batch [{batch_idx+1}/{len(train_loader)}] | Loss: {loss.item():.4f}")
                
        epoch_train_loss = running_loss / total_train
        epoch_train_acc = (correct_train / total_train) * 100
        
        # --- Validation Phase ---
        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                running_val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
                
                val_preds.extend(predicted.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())
                
        epoch_val_loss = running_val_loss / total_val
        epoch_val_acc = (correct_val / total_val) * 100
        
        # Calculate macro F1 (handles class imbalance evaluation properly)
        epoch_val_f1 = f1_score(val_targets, val_preds, average='macro')
        
        # Update scheduler
        scheduler.step(epoch_val_f1)
        
        # Save metrics history
        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)
        history["val_f1"].append(epoch_val_f1)
        
        print(f"\n--- Epoch {epoch+1} Summary ---")
        print(f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.2f}%")
        print(f"Val Loss:   {epoch_val_loss:.4f} | Val Acc:   {epoch_val_acc:.2f}% | Val F1 (Macro): {epoch_val_f1:.4f}")
        
        # Checkpoint based on F1-Score
        if epoch_val_f1 > best_val_f1:
            best_val_f1 = epoch_val_f1
            patience_counter = 0
            model_save_path = os.path.join(args.output_dir, args.model_save_name)
            torch.save(model.state_dict(), model_save_path)
            print(f"[Best Checkpoint] New best Val F1 ({best_val_f1:.4f})! Saved checkpoint to {model_save_path}")
        else:
            patience_counter += 1
            print(f"No improvement in Val F1. Early stopping counter: {patience_counter}/{args.patience}")
            
        # Early Stopping check
        if patience_counter >= args.patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break
            
    # Save training history as JSON
    history_path = os.path.join(args.output_dir, "training_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4)
    print(f"Saved history stats to {history_path}")
    
    # 4. Generate & Save Plots
    epochs_trained = range(1, len(history["train_loss"]) + 1)
    
    # Loss plot
    plt.figure(figsize=(10, 5))
    plt.plot(epochs_trained, history["train_loss"], 'bo-', label='Training Loss')
    plt.plot(epochs_trained, history["val_loss"], 'ro-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    loss_plot_path = os.path.join(args.output_dir, 'loss_curves.png')
    plt.savefig(loss_plot_path)
    plt.close()
    
    # Accuracy & F1 plot
    plt.figure(figsize=(10, 5))
    plt.plot(epochs_trained, history["train_acc"], 'b--', label='Training Acc (%)')
    plt.plot(epochs_trained, history["val_acc"], 'r--', label='Validation Acc (%)')
    plt.plot(epochs_trained, [f * 100 for f in history["val_f1"]], 'g-', label='Val F1 (Macro x100)')
    plt.title('Accuracy and F1 Score')
    plt.xlabel('Epochs')
    plt.ylabel('Percentage')
    plt.legend()
    plt.grid(True)
    metrics_plot_path = os.path.join(args.output_dir, 'metrics_curves.png')
    plt.savefig(metrics_plot_path)
    plt.close()
    
    print(f"Plots saved to {args.output_dir}/")
    print("Training Complete!")
    return model_save_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Cervical Cancer Model")
    parser.add_argument("--data_dir", type=str, default="D:/MSC AI/Split_Data", help="Path to Split_Data directory")
    parser.add_argument("--output_dir", type=str, default="D:/MSC AI/cervical_classifier/outputs", help="Directory to save weights and logs")
    parser.add_argument("--model_save_name", type=str, default="best_cervical_model.pth", help="Filename of the saved checkpoint")
    parser.add_argument("--model_name", type=str, default="efficientnet_b0", help="Model backbone (efficientnet_b0, resnet18, resnet50)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="DataLoader batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="L2 regularization weight decay")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience epochs")
    parser.add_argument("--freeze_backbone", action="store_true", help="Freeze the model's pretrained backbone layers")
    parser.add_argument("--no_pretrained", action="store_true", help="Train from scratch (do not use ImageNet pretrained weights)")
    
    args = parser.parse_args()
    train_model(args)
