import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import cv2
import matplotlib.pyplot as plt

from dataset import get_transforms
from model import get_model
from explainability import create_explainability_overlay

def run_prediction(args):
    # Setup directories
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Image loading and transform configuration
    if not os.path.exists(args.image_path):
        raise FileNotFoundError(f"Target image not found: {args.image_path}")
        
    _, val_test_transform = get_transforms(img_size=(224, 224))
    
    # 2. Build and load model
    classes = ['CIN1', 'CIN2', 'CIN3', 'Normal']
    model = get_model(
        num_classes=len(classes),
        model_name=args.model_name,
        pretrained=False
    )
    
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model checkpoint weights not found: {args.model_path}")
        
    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    # 3. Generate XAI and localization overlays
    print(f"\nProcessing image: {args.image_path}...")
    result = create_explainability_overlay(
        image_path=args.image_path,
        model=model,
        transform=val_test_transform,
        model_name=args.model_name,
        device=device,
        threshold=args.threshold
    )
    
    pred_class = classes[result["pred_idx"]]
    confidence = result["confidence"]
    bbox = result["bbox"]
    
    print(f"\nPredicted Diagnosis : {pred_class}  |  Confidence: {confidence * 100:.2f}%")
    
    # 4. Swede Score Calculation
    # ------------------------------------------------------------------
    # COMPONENT SOURCE LEGEND:
    #   [RULE-BASED]    : Value looked up from a fixed class-to-score table
    #                     keyed on the classifier's predicted grade label.
    #                     These values are NOT independently estimated from
    #                     image pixels; they reflect clinical prototypes per
    #                     CIN grade as defined in colposcopy guidelines.
    #   [GRAD-CAM EST.] : Value dynamically estimated from the spatial area
    #                     coverage of the thresholded Grad-CAM activation
    #                     mask — the only image-derived Swede component.
    # ------------------------------------------------------------------

    # --- [GRAD-CAM EST.] Lesion Size Score ---
    # Computed from the proportion of the image covered by the thresholded
    # Grad-CAM heatmap (threshold tau = args.threshold).
    heatmap = result["heatmap"]
    mask = heatmap > args.threshold
    lesion_pct = (np.sum(mask) / heatmap.size) * 100

    suggested_size = 0
    if lesion_pct > 15:
        suggested_size = 2   # >15% coverage  -> score 2
    elif lesion_pct > 5:
        suggested_size = 1   # 5-15% coverage -> score 1
    # else:                  # <5%  coverage  -> score 0

    # --- [RULE-BASED] Acetowhiteness, Margins, Vessels, Iodine ---
    # Fixed lookup table mapping predicted class label to prototype scores.
    # Source: standard colposcopy Swede Score criteria per CIN grade.
    SWEDE_LOOKUP = {
        'Normal': (0, 0, 0, 0),
        'CIN1':   (1, 0, 0, 1),
        'CIN2':   (1, 1, 1, 1),
        'CIN3':   (2, 2, 2, 2),
    }
    aceto, margin, vessels, iodine = SWEDE_LOOKUP.get(pred_class, (0, 0, 0, 0))

    total_swede_score = aceto + margin + vessels + iodine + suggested_size

    # Risk categorisation
    if total_swede_score >= 7:
        risk_level = "High Risk (HSIL+): Urgent biopsy and treatment evaluation required."
        risk_tag   = "HIGH"
    elif total_swede_score >= 5:
        risk_level = "Intermediate Risk (CIN2+): Colposcopic biopsy highly recommended."
        risk_tag   = "INTERMEDIATE"
    else:
        risk_level = "Low Risk: Likely Normal or low-grade. Routine follow-up per protocol."
        risk_tag   = "LOW"

    # Build the console report
    report_lines = [
        "=" * 50,
        "            DIAGNOSTIC REPORT",
        "=" * 50,
        f"Predicted Diagnosis : {pred_class}",
        f"Confidence Score    : {confidence * 100:.2f}%",
        "",
        "--- Automated Swede Score ---",
        f"  Grad-CAM Lesion Coverage  : {lesion_pct:.1f}% of image",
        "",
        "  Swede Component                 Score   Source",
        "  " + "-" * 46,
        f"  Acetowhiteness (0-2)         :   {aceto}     [RULE-BASED]",
        f"  Margins / Surface  (0-2)     :   {margin}     [RULE-BASED]",
        f"  Vessel Patterns    (0-2)     :   {vessels}     [RULE-BASED]",
        f"  Iodine Staining    (0-2)     :   {iodine}     [RULE-BASED]",
        f"  Lesion Size Score  (0-2)     :   {suggested_size}     [GRAD-CAM EST.]",
        "  " + "-" * 46,
        f"  Total Swede Score            :   {total_swede_score} / 10",
        "",
        f"Risk Level          : {risk_tag}",
        f"Action Protocol     : {risk_level}",
    ]

    if bbox:
        x, y, w, h = bbox
        report_lines += [
            "",
            "--- Cancer/Lesion Localization ---",
            f"  Bounding Box: [X: {x}, Y: {y}, Width: {w}, Height: {h}]",
        ]
    else:
        report_lines += [
            "",
            "--- Cancer/Lesion Localization ---",
            "  No high-concentration lesion area detected above threshold.",
        ]

    report_lines.append("=" * 50)
    report_text = "\n".join(report_lines)
    print("\n" + report_text + "\n")

    # Save plain-text clinical report
    report_filename = f"report_{os.path.splitext(os.path.basename(args.image_path))[0]}.txt"
    report_path = os.path.join(args.output_dir, report_filename)
    with open(report_path, "w") as rf:
        rf.write(report_text + "\n")
    print(f"Saved clinical report to: {report_path}")
    
    # 5. Save multi-panel visualization (4-panel with Swede score breakdown)
    fig = plt.figure(figsize=(22, 6))
    fig.suptitle(
        f"CerviScan AI — {pred_class}  |  Confidence: {confidence * 100:.2f}%  |  "
        f"Swede Score: {total_swede_score}/10  [{risk_tag} RISK]",
        fontsize=13, fontweight='bold', y=1.01
    )

    ax0 = fig.add_subplot(1, 4, 1)
    ax0.imshow(result["original"])
    ax0.set_title("Original Image", fontsize=11)
    ax0.axis('off')

    ax1 = fig.add_subplot(1, 4, 2)
    ax1.imshow(result["heatmap"], cmap='jet')
    ax1.set_title(f"Grad-CAM Heatmap\nCoverage: {lesion_pct:.1f}%", fontsize=11)
    ax1.axis('off')

    ax2 = fig.add_subplot(1, 4, 3)
    ax2.imshow(result["annotated"])
    ax2.set_title(f"Lesion Localization", fontsize=11)
    ax2.axis('off')

    # Standalone Panel: Save Swede score bar chart separately
    fig_bar, ax_bar = plt.subplots(figsize=(6, 5))
    components      = ['Aceto\nwhite', 'Margins\nSurface', 'Vessel\nPattern', 'Iodine\nStaining', 'Lesion\nSize']
    scores          = [aceto, margin, vessels, iodine, suggested_size]
    source_labels   = ['Rule-\nbased', 'Rule-\nbased', 'Rule-\nbased', 'Rule-\nbased', 'Grad-CAM\nEst.']
    bar_colors      = ['#2e7d32', '#2e7d32', '#2e7d32', '#2e7d32', '#d84315']
    bars_std = ax_bar.bar(components, scores, color=bar_colors, edgecolor='white', linewidth=1.2)
    ax_bar.set_ylim(0, 2.5)
    ax_bar.set_yticks([0, 1, 2])
    ax_bar.set_title(f"Swede Score Components (Total: {total_swede_score}/10)", fontsize=12, fontweight='bold')
    ax_bar.set_ylabel("Score")
    for bar, src, sc in zip(bars_std, source_labels, scores):
        ax_bar.text(bar.get_x() + bar.get_width() / 2, sc + 0.05, str(sc),
                 ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax_bar.text(bar.get_x() + bar.get_width() / 2, -0.35, src,
                 ha='center', va='top', fontsize=7, color='grey')
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2e7d32', label='Rule-based (class label)'),
        Patch(facecolor='#d84315', label='Grad-CAM estimated'),
    ]
    ax_bar.legend(handles=legend_elements, loc='upper right', fontsize=8)
    ax_bar.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    bar_output_path = os.path.join(args.output_dir, f"swede_bar_chart_{os.path.splitext(os.path.basename(args.image_path))[0]}.png")
    plt.savefig(bar_output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved standalone Swede bar chart to: {bar_output_path}")

    # 5. Save multi-panel visualization (4-panel with Swede score breakdown)
    fig = plt.figure(figsize=(22, 6))
    fig.suptitle(
        f"CerviScan AI — {pred_class}  |  Confidence: {confidence * 100:.2f}%  |  "
        f"Swede Score: {total_swede_score}/10  [{risk_tag} RISK]",
        fontsize=13, fontweight='bold', y=1.01
    )

    ax0 = fig.add_subplot(1, 4, 1)
    ax0.imshow(result["original"])
    ax0.set_title("Original Image", fontsize=11)
    ax0.axis('off')

    ax1 = fig.add_subplot(1, 4, 2)
    ax1.imshow(result["heatmap"], cmap='jet')
    ax1.set_title(f"Grad-CAM Heatmap\nCoverage: {lesion_pct:.1f}%", fontsize=11)
    ax1.axis('off')

    ax2 = fig.add_subplot(1, 4, 3)
    ax2.imshow(result["annotated"])
    ax2.set_title(f"Lesion Localization", fontsize=11)
    ax2.axis('off')

    # Panel 4: Swede score bar chart
    ax3 = fig.add_subplot(1, 4, 4)
    bars = ax3.bar(components, scores, color=bar_colors, edgecolor='white', linewidth=1.2)
    ax3.set_ylim(0, 2.5)
    ax3.set_yticks([0, 1, 2])
    ax3.set_title(f"Swede Score Components\nTotal: {total_swede_score}/10", fontsize=11)
    ax3.set_ylabel("Score")
    for bar, src, sc in zip(bars, source_labels, scores):
        ax3.text(bar.get_x() + bar.get_width() / 2, sc + 0.05, str(sc),
                 ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax3.text(bar.get_x() + bar.get_width() / 2, -0.35, src,
                 ha='center', va='top', fontsize=6.5, color='grey')
    ax3.legend(handles=legend_elements, loc='upper right', fontsize=7)
    ax3.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    output_filename = f"prediction_{os.path.basename(args.image_path)}"
    output_path = os.path.join(args.output_dir, output_filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved explainability visual analysis to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict Cervical Cancer stage and localize lesion region")
    parser.add_argument("--image_path", type=str, required=True, help="Path to cervical/colposcopy image file")
    parser.add_argument("--model_path", type=str, default="D:/MSC AI/cervical_model.pth", help="Path to trained .pth weights")
    parser.add_argument("--model_name", type=str, default="efficientnet_b0", help="Model architecture")
    parser.add_argument("--output_dir", type=str, default="D:/MSC AI/cervical_classifier/outputs_corrected", help="Directory to save corrected outputs (image + report)")
    parser.add_argument("--threshold", type=float, default=0.4, help="Activation threshold for lesion boundary segmentation")
    
    args = parser.parse_args()
    run_prediction(args)
