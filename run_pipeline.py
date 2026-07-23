import os
import sys
import argparse
import subprocess
import glob

def run_cmd(cmd_list):
    """Utility to run commands and stream output."""
    print(f"\nRunning command: {' '.join(cmd_list)}")
    result = subprocess.run(cmd_list, stdout=sys.stdout, stderr=sys.stderr)
    if result.returncode != 0:
        print(f"Command failed with exit code {result.returncode}")
        sys.exit(result.returncode)

def main():
    parser = argparse.ArgumentParser(description="Run the entire Cervical Cancer Classification & XAI pipeline")
    parser.add_argument("--data_dir", type=str, default="D:/MSC AI/Split_Data", help="Path to Split_Data directory")
    parser.add_argument("--output_dir", type=str, default="D:/MSC AI/cervical_classifier/outputs", help="Directory for output files")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--model_name", type=str, default="efficientnet_b0", help="Model backbone")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience")
    parser.add_argument("--freeze_backbone", action="store_true", help="Freeze pretrained backbone")
    parser.add_argument("--sample_image", type=str, default="", help="Optional path to a test image for explainability demo")
    
    args = parser.parse_args()
    
    # Paths for scripts
    python_exe = sys.executable
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    train_script = os.path.join(script_dir, "train.py")
    eval_script = os.path.join(script_dir, "evaluate.py")
    predict_script = os.path.join(script_dir, "predict.py")
    
    model_save_name = "best_cervical_model.pth"
    model_path = os.path.join(args.output_dir, model_save_name)
    
    print("=" * 60)
    print("      CERVICAL CANCER CLASSIFICATION & XAI PIPELINE      ")
    print("=" * 60)
    
    # Step 1: Train the model
    print("\n>>> STEP 1: TRAINING MODEL")
    train_cmd = [
        python_exe, train_script,
        "--data_dir", args.data_dir,
        "--output_dir", args.output_dir,
        "--model_save_name", model_save_name,
        "--model_name", args.model_name,
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--lr", str(args.lr),
        "--patience", str(args.patience)
    ]
    if args.freeze_backbone:
        train_cmd.append("--freeze_backbone")
    run_cmd(train_cmd)
    
    # Step 2: Evaluate the model on test set
    print("\n>>> STEP 2: RUNNING TEST SET EVALUATION")
    eval_cmd = [
        python_exe, eval_script,
        "--data_dir", args.data_dir,
        "--model_path", model_path,
        "--model_name", args.model_name,
        "--output_dir", args.output_dir,
        "--batch_size", str(args.batch_size)
    ]
    run_cmd(eval_cmd)
    
    # Step 3: Explainability and Localization Inference demo on sample image
    print("\n>>> STEP 3: EXPLAINABILITY & LESION LOCALIZATION DEMO")
    
    image_path = args.sample_image
    if not image_path:
        # Automatically find an image from test set to demo
        test_images = []
        test_dir = os.path.join(args.data_dir, "test")
        if os.path.exists(test_dir):
            for ext in ('*.jpg', '*.jpeg', '*.png'):
                test_images.extend(glob.glob(os.path.join(test_dir, "**", ext), recursive=True))
                
        if len(test_images) > 0:
            # Prefer a CIN case if available to show interesting contours
            cin_images = [img for img in test_images if "normal" not in img.lower()]
            image_path = cin_images[0] if len(cin_images) > 0 else test_images[0]
            print(f"Auto-selected test image for explainability demo: {image_path}")
        else:
            print("Warning: No test images found. Skipping explainability demo step.")
            
    if image_path:
        predict_cmd = [
            python_exe, predict_script,
            "--image_path", image_path,
            "--model_path", model_path,
            "--model_name", args.model_name,
            "--output_dir", args.output_dir,
            "--threshold", "0.4"
        ]
        run_cmd(predict_cmd)
        
    print("\n" + "=" * 60)
    print("                  PIPELINE WORKFLOW FINISHED                  ")
    print("=" * 60)

if __name__ == "__main__":
    main()
