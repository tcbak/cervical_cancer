import os
import json
import time
import random
import argparse
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import (
    f1_score,
    accuracy_score,
)

from dataset import get_dataloaders
from model import (
    get_model,
    get_input_size,
    get_model_info
)


# Reproducibility


def set_seed(seed=42):
    """
    Makes experiments reproducible.
    """

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



# Helper Functions


def count_parameters(model):
    """
    Returns total and trainable parameters.
    """

    total = sum(p.numel() for p in model.parameters())

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    return total, trainable


def save_json(data, filename):

    with open(filename, "w") as f:
        json.dump(data, f, indent=4)



# Experiment Folder Creation


def create_experiment_folders(root_dir, model_name):

    experiment_dir = os.path.join(root_dir, model_name)

    folders = {

        "root": experiment_dir,

        "checkpoints": os.path.join(
            experiment_dir,
            "checkpoints"
        ),

        "plots": os.path.join(
            experiment_dir,
            "plots"
        ),

        "logs": os.path.join(
            experiment_dir,
            "logs"
        ),

        "evaluation": os.path.join(
            experiment_dir,
            "evaluation"
        ),

        "benchmark": os.path.join(
            experiment_dir,
            "benchmark"
        )
    }

    for folder in folders.values():
        os.makedirs(folder, exist_ok=True)

    return folders



# Training Function


def train_model(args):

    set_seed(42)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    
    print("Cervical Cancer Benchmark Training")
    
    print(f"Device : {device}")

    model_info = get_model_info(args.model_name)

    print(f"Model  : {model_info['display_name']}")
    print(f"Type   : {model_info['category']}")

    folders = create_experiment_folders(
        args.output_dir,
        args.model_name
    )

    img_size = get_input_size(
        args.model_name
    )

    print(f"Input Size : {img_size}")

    experiment_info = {

        "model_name": args.model_name,

        "display_name": model_info["display_name"],

        "category": model_info["category"],

        "epochs": args.epochs,

        "batch_size": args.batch_size,

        "learning_rate": args.lr,

        "weight_decay": args.weight_decay,

        "patience": args.patience,

        "pretrained": not args.no_pretrained,

        "freeze_backbone": args.freeze_backbone,

        "input_size": img_size,

        "device": str(device),

        "date": str(datetime.now())
    }

    save_json(

        experiment_info,

        os.path.join(

            folders["logs"],

            "experiment.json"

        )

    )

    print("\nLoading Dataset...")

    train_loader, val_loader, _, class_names, class_weights = get_dataloaders(

        data_dir=args.data_dir,

        batch_size=args.batch_size,

        img_size=img_size

    )

    print("Classes")

    print(class_names)

    print("Loss Weights")

    print(class_weights)

    num_classes = len(class_names)

    
    # Build Model
    

    print("\nBuilding Model...")

    model = get_model(
        num_classes=num_classes,
        model_name=args.model_name,
        pretrained=not args.no_pretrained,
        freeze_backbone=args.freeze_backbone
    )

    model = model.to(device)

    total_params, trainable_params = count_parameters(model)

    print(f"Total Parameters      : {total_params:,}")
    print(f"Trainable Parameters  : {trainable_params:,}")

    model_summary = {
        "model_name": args.model_name,
        "display_name": model_info["display_name"],
        "category": model_info["category"],
        "input_size": img_size,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params
        
    }

    save_json(
        model_summary,
        os.path.join(
            folders["logs"],
            "model_summary.json"
        )
    )

    
    # Loss Function
    

    print("\nInitializing Loss Function...")

    loss_weights = class_weights.to(device)

    criterion = nn.CrossEntropyLoss(
        weight=loss_weights
    )

    
    # Optimizer
   

    optimizer = optim.Adam(

        model.parameters(),

        lr=args.lr,

        weight_decay=args.weight_decay

    )

    
    # Learning Rate Scheduler
   

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(

        optimizer,

        mode="max",

        factor=0.5,

        patience=2

    )

    
    # Checkpoint Paths
   

    best_checkpoint_path = os.path.join(
        folders["checkpoints"],
        "best_model.pth"
    )

    last_checkpoint_path = os.path.join(
        folders["checkpoints"],
        "last_model.pth"
    )
    
    # Training History
    

    history = {

        "train_loss": [],

        "train_accuracy": [],

        "validation_loss": [],

        "validation_accuracy": [],

        "validation_f1": []

    }

    best_val_f1 = 0.0

    best_epoch = 0

    patience_counter = 0

    
    # Training Timer
   

    training_start = time.time()

    epoch_times = []

    
    print("Starting Training")

    
    # Training Loop
    

    for epoch in range(args.epochs):

        
        print(f"Epoch {epoch + 1}/{args.epochs}")
        

        epoch_start = time.time()

       
        # Training Phase
        
        model.train()

        running_loss = 0.0
        running_total = 0

        train_predictions = []
        train_targets = []

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

          
            # Statistics
            
            running_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs, 1)

            running_total += labels.size(0)

            train_predictions.extend(
                predicted.cpu().numpy()
            )

            train_targets.extend(
                labels.cpu().numpy()
            )

        
        # Training Metrics
        

        train_loss = running_loss / running_total

        train_accuracy = accuracy_score(
            train_targets,
            train_predictions
        ) * 100

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)

        print(f"Training Loss      : {train_loss:.4f}")
        print(f"Training Accuracy  : {train_accuracy:.2f}%")

        
        # Validation Phase
        

        model.eval()

        val_running_loss = 0.0
        val_predictions = []
        val_targets = []

        with torch.no_grad():

            for images, labels in val_loader:

                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)

                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * images.size(0)

                _, predicted = torch.max(outputs, 1)

                val_predictions.extend(
                    predicted.cpu().numpy()
                )

                val_targets.extend(
                    labels.cpu().numpy()
                )

        
        # Validation Metrics
        

        validation_loss = val_running_loss / len(val_loader.dataset)

        validation_accuracy = accuracy_score(
            val_targets,
            val_predictions
        ) * 100

        validation_f1 = f1_score(
            val_targets,
            val_predictions,
            average="macro"
        )

        history["validation_loss"].append(validation_loss)
        history["validation_accuracy"].append(validation_accuracy)
        history["validation_f1"].append(validation_f1)

        print(f"Validation Loss     : {validation_loss:.4f}")
        print(f"Validation Accuracy : {validation_accuracy:.2f}%")
        print(f"Validation Macro F1 : {validation_f1:.4f}")

      
        # Learning Rate Scheduler
        

        scheduler.step(validation_f1)

        
        # Save Best Model
        

        if validation_f1 > best_val_f1:

            best_val_f1 = validation_f1
            best_epoch = epoch + 1
            patience_counter = 0

            torch.save(
                model.state_dict(),
                best_checkpoint_path
            )

            print("\n✓ New Best Model Saved!")
            print(f"Best Validation F1 : {best_val_f1:.4f}")
            print(f"Best Checkpoint Saved To : {best_checkpoint_path}")

        else:

            patience_counter += 1

            print(
                f"\nNo Improvement "
                f"({patience_counter}/{args.patience})"
            )
        # Save latest model every epoch
        torch.save(
            model.state_dict(),
            last_checkpoint_path
        )
        # Epoch Timing
       

        epoch_time = time.time() - epoch_start

        epoch_times.append(epoch_time)

        print(f"Epoch Time : {epoch_time:.2f} sec")

        
        # Early Stopping
        

        if patience_counter >= args.patience:

            print("\nEarly Stopping Triggered.")
            break

    
    # Training Finished
    

    training_time = time.time() - training_start
    average_epoch_time = np.mean(epoch_times) if epoch_times else 0

    
    print("Training Completed")
    

    print(f"Best Epoch               : {best_epoch}")
    print(f"Best Validation F1       : {best_val_f1:.4f}")
    print(f"Total Training Time      : {training_time/60:.2f} minutes")
    print(f"Average Epoch Time       : {average_epoch_time:.2f} seconds")
    print(f"Best Checkpoint : {best_checkpoint_path}")
    print(f"Last Checkpoint : {last_checkpoint_path}")  

   
    # Save Training History
    

    history_path = os.path.join(
        folders["logs"],
        "training_history.json"
    )

    save_json(history, history_path)

    #
    # Save Training Summary
    

    training_summary = {

        "model_name": args.model_name,

        "best_epoch": best_epoch,

        "best_validation_f1": float(best_val_f1),

        "training_time_seconds": float(training_time),

        "training_time_minutes": float(training_time / 60),

        "average_epoch_time_seconds": float(average_epoch_time),

        "total_parameters": total_params,

        "trainable_parameters": trainable_params,

        "total_epochs_completed": len(history["train_loss"])

    }

    save_json(

        training_summary,

        os.path.join(

            folders["logs"],

            "training_summary.json"

        )

    )

    
    # Plot Loss Curve
    

    plt.figure(figsize=(8,5))

    plt.plot(
        history["train_loss"],
        label="Training Loss"
    )

    plt.plot(
        history["validation_loss"],
        label="Validation Loss"
    )

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.title("Loss Curve")

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    plt.savefig(

        os.path.join(

            folders["plots"],

            "loss_curve.png"

        )

    )

    plt.close()

    
    # Plot Accuracy Curve
    

    plt.figure(figsize=(8,5))

    plt.plot(
        history["train_accuracy"],
        label="Training Accuracy"
    )

    plt.plot(
        history["validation_accuracy"],
        label="Validation Accuracy"
    )

    plt.xlabel("Epoch")

    plt.ylabel("Accuracy (%)")

    plt.title("Accuracy Curve")

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    plt.savefig(

        os.path.join(

            folders["plots"],

            "accuracy_curve.png"

        )

    )

    plt.close()

    
    # Plot Validation F1
    

    plt.figure(figsize=(8,5))

    plt.plot(

        history["validation_f1"],

        color="green",

        linewidth=2

    )

    plt.xlabel("Epoch")

    plt.ylabel("Macro F1")

    plt.title("Validation Macro F1")

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(

        os.path.join(

            folders["plots"],

            "validation_f1.png"

        )

    )

    plt.close()

    print("\nTraining history saved.")

    print("Plots saved.")

    print("Training summary saved.")

    return best_checkpoint_path


# Main


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Train Cervical Cancer Classification Model"
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default="Split_Data",
        help="Dataset directory"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="benchmark_results",
        help="Output directory"
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="efficientnet_b0",
        help="Model architecture"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=20
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.0001
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=5
    )

    parser.add_argument(
        "--freeze_backbone",
        action="store_true"
    )

    parser.add_argument(
        "--no_pretrained",
        action="store_true"
    )

    args = parser.parse_args()

    train_model(args)