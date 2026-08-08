import os
import json
import time
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

from dataset import get_dataloaders
from model import (
    get_model,
    get_input_size,
    get_model_info
)

# ==========================================================
# Helper Functions
# ==========================================================

def save_json(data, filename):
    """
    Saves a dictionary as a JSON file.
    """
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)


def create_evaluation_folders(output_root):
    """
    Creates the folders required for evaluation outputs.
    """

    folders = {

        "root": output_root,

        "reports": os.path.join(
            output_root,
            "reports"
        ),

        "plots": os.path.join(
            output_root,
            "plots"
        ),

        "metrics": os.path.join(
            output_root,
            "metrics"
        )

    }

    for folder in folders.values():
        os.makedirs(folder, exist_ok=True)

    return folders


# ==========================================================
# Sensitivity / Specificity
# ==========================================================

def per_class_sensitivity_specificity(cm):
    """
    Computes one-vs-rest sensitivity and specificity for each class.
    """

    n_classes = cm.shape[0]

    sensitivity = []

    specificity = []

    total = cm.sum()

    for i in range(n_classes):

        tp = cm[i, i]

        fn = cm[i, :].sum() - tp

        fp = cm[:, i].sum() - tp

        tn = total - tp - fn - fp

        sens = tp / (tp + fn) if (tp + fn) > 0 else 0

        spec = tn / (tn + fp) if (tn + fp) > 0 else 0

        sensitivity.append(sens)

        specificity.append(spec)

    return sensitivity, specificity

def evaluate_model(args):

    # ==========================================================
    # Device
    # ==========================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 80)
    print("Model Evaluation")
    print("=" * 80)

    print(f"Device : {device}")

    # ==========================================================
    # Model Information
    # ==========================================================

    model_info = get_model_info(
        args.model_name
    )

    print(f"Model : {model_info['display_name']}")
    print(f"Category : {model_info['category']}")

    # ==========================================================
    # Create Output Folders
    # ==========================================================

    folders = create_evaluation_folders(
        args.output_dir
    )

    # ==========================================================
    # Input Size
    # ==========================================================

    img_size = get_input_size(
        args.model_name
    )

    print(f"Input Size : {img_size}")

    # ==========================================================
    # Dataset
    # ==========================================================

    print("\nLoading Test Dataset...")

    _, _, test_loader, class_names, class_weights = get_dataloaders(

        data_dir=args.data_dir,

        batch_size=args.batch_size,

        img_size=img_size

    )

    num_classes = len(class_names)

    print("Classes")

    print(class_names)

    # ==========================================================
    # Build Model
    # ==========================================================

    print("\nBuilding Model...")

    model = get_model(

        num_classes=num_classes,

        model_name=args.model_name,

        pretrained=False

    )

    # ==========================================================
    # Load Checkpoint
    # ==========================================================

    if not os.path.exists(args.model_path):

        raise FileNotFoundError(

            f"Checkpoint not found:\n{args.model_path}"

        )

    print("\nLoading trained checkpoint...")

    print(args.model_path)

    state_dict = torch.load(

        args.model_path,

        map_location=device

    )

    model.load_state_dict(state_dict)

    model = model.to(device)

    model.eval()

    # ==========================================================
    # Loss Function
    # ==========================================================

    criterion = nn.CrossEntropyLoss(

        weight=class_weights.to(device)

    )

    # ==========================================================
    # Variables
    # ==========================================================

    evaluation_start = time.time()

    test_loss = 0.0

    total = 0

    correct = 0

    all_predictions = []

    all_targets = []

    all_probabilities = []

    print("\nStarting Evaluation...\n")

    # ==========================================================
    # Evaluation Loop
    # ==========================================================

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)

            labels = labels.to(device)

            # -----------------------------
            # Forward Pass
            # -----------------------------

            outputs = model(images)

            loss = criterion(outputs, labels)

            test_loss += loss.item() * images.size(0)

            # -----------------------------
            # Softmax Probabilities
            # -----------------------------

            probabilities = F.softmax(outputs, dim=1)

            predictions = torch.argmax(probabilities, dim=1)

            # -----------------------------
            # Accuracy Statistics
            # -----------------------------

            total += labels.size(0)

            correct += (predictions == labels).sum().item()

            # -----------------------------
            # Store Results
            # -----------------------------

            all_predictions.extend(

                predictions.cpu().numpy()

            )

            all_targets.extend(

                labels.cpu().numpy()

            )

            all_probabilities.extend(

                probabilities.cpu().numpy()

            )

    # ==========================================================
    # Evaluation Finished
    # ==========================================================

    evaluation_time = time.time() - evaluation_start

    all_predictions = np.array(all_predictions)

    all_targets = np.array(all_targets)

    all_probabilities = np.array(all_probabilities)

        # ==========================================================
    # Calculate Metrics
    # ==========================================================

    average_test_loss = test_loss / total

    accuracy = accuracy_score(

        all_targets,

        all_predictions

    )

    precision = precision_score(

        all_targets,

        all_predictions,

        average="macro",

        zero_division=0

    )

    recall = recall_score(

        all_targets,

        all_predictions,

        average="macro",

        zero_division=0

    )

    macro_f1 = f1_score(

        all_targets,

        all_predictions,

        average="macro"

    )

    weighted_f1 = f1_score(

        all_targets,

        all_predictions,

        average="weighted"

    )

    # ==========================================================
    # AUC
    # ==========================================================

    try:

        macro_auc = roc_auc_score(

            all_targets,

            all_probabilities,

            average="macro",

            multi_class="ovr"

        )

    except Exception:

        macro_auc = None

    # ==========================================================
    # Confusion Matrix
    # ==========================================================

    conf_matrix = confusion_matrix(

        all_targets,

        all_predictions

    )

    normalized_cm = (

        conf_matrix.astype(float)

        / conf_matrix.sum(axis=1)[:, np.newaxis]

    )

    sensitivity, specificity = per_class_sensitivity_specificity(

        conf_matrix

    )

        # ==========================================================
    # Save Confusion Matrix CSV
    # ==========================================================

    cm_df = pd.DataFrame(

        conf_matrix,

        index=class_names,

        columns=class_names

    )

    cm_df.to_csv(

        os.path.join(

            folders["metrics"],

            "confusion_matrix.csv"

        )

    )
    # ==========================================================
    # Console Output
    # ==========================================================

    print("\n")

    print("=" * 80)

    print("Evaluation Summary")

    print("=" * 80)

    print(f"Average Test Loss : {average_test_loss:.4f}")

    print(f"Accuracy          : {accuracy*100:.2f}%")

    print(f"Precision         : {precision:.4f}")

    print(f"Recall            : {recall:.4f}")

    print(f"Macro F1          : {macro_f1:.4f}")

    print(f"Weighted F1       : {weighted_f1:.4f}")

    if macro_auc is not None:

        print(f"Macro AUC         : {macro_auc:.4f}")

    print(f"Evaluation Time   : {evaluation_time:.2f} seconds")

    print("=" * 80)

    # ==========================================================
    # Classification Report
    # ==========================================================

    report = classification_report(

        all_targets,

        all_predictions,

        target_names=class_names,

        digits=4

    )

    print("\nClassification Report\n")

    print(report)

    report_path = os.path.join(

        folders["reports"],

        "classification_report.txt"

    )

    with open(report_path, "w") as f:

        f.write(report)

    # ==========================================================
    # Confusion Matrix Figure
    # ==========================================================

    plt.figure(figsize=(8,6))

    plt.imshow(

        normalized_cm,

        interpolation="nearest",

        cmap=plt.cm.Blues

    )

    plt.title("Normalized Confusion Matrix")

    plt.colorbar()

    tick_marks = np.arange(len(class_names))

    plt.xticks(

        tick_marks,

        class_names,

        rotation=45

    )

    plt.yticks(

        tick_marks,

        class_names

    )

    threshold = normalized_cm.max() / 2

    for i in range(normalized_cm.shape[0]):

        for j in range(normalized_cm.shape[1]):

            plt.text(

                j,

                i,

                f"{normalized_cm[i,j]:.2f}",

                ha="center",

                color="white" if normalized_cm[i,j] > threshold else "black"

            )

    plt.tight_layout()

    plt.ylabel("True Label")

    plt.xlabel("Predicted Label")

    plt.savefig(

        os.path.join(

            folders["plots"],

            "confusion_matrix.png"

        ),

        dpi=300,

        bbox_inches="tight"

    )

    plt.close()

    # ==========================================================
    # Save Metrics JSON
    # ==========================================================

    metrics = {

        "model_name": args.model_name,

        "num_classes": num_classes,

        "num_test_images": total,

        "accuracy": round(float(accuracy),4),

        "precision": round(float(precision),4),

        "recall": round(float(recall),4),

        "macro_f1": round(float(macro_f1),4),

        "weighted_f1": round(float(weighted_f1),4),

        "macro_auc": round(float(macro_auc),4) if macro_auc is not None else None,

        "test_loss": round(float(average_test_loss),4),

        "evaluation_time_seconds": round(float(evaluation_time),2)

    }

    save_json(

        metrics,

        os.path.join(

            folders["metrics"],

            "metrics.json"

        )

    )

    # ==========================================================
    # Per-Class Metrics
    # ==========================================================

    per_class = {}

    for i, cls in enumerate(class_names):

        per_class[cls] = {

            "sensitivity": round(float(sensitivity[i]),4),

            "specificity": round(float(specificity[i]),4)

        }

    save_json(

        per_class,

        os.path.join(

            folders["metrics"],

            "per_class_metrics.json"

        )

    )

    print("\n")

    print("Evaluation Completed Successfully")

    print(f"Results saved to:")

    print(f"   {folders['root']}")


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Evaluate Cervical Cancer Classification Model"
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default="Split_Data"
    )

    parser.add_argument(
        "--model_path",
        type=str,
        required=True
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="efficientnet_b0"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="benchmark_results"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32
    )

    args = parser.parse_args()

    evaluate_model(args)