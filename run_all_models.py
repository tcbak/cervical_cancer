import os
import sys
import subprocess
import time

# ==========================================================
# Models to Train
# ==========================================================

MODELS = [
    "efficientnet_b0",

    "mobilenet_v2",

    "shufflenet_v2_x1_0",

    "squeezenet1_1",

    "densenet121",

    "resnet18",

    "resnet34",

    "resnet50",

    "vgg16"


]

RESULTS_ROOT = "benchmark_results"

DATASET_PATH = "Split_Data"

# ==========================================================
# Train Model
# ==========================================================

def train_model(model_name):

    
    print(f"TRAINING {model_name}")
    

    subprocess.run([

        sys.executable,

        "train.py",

        "--model_name",

        model_name,

        "--data_dir",

        DATASET_PATH,

        "--output_dir",

        RESULTS_ROOT

    ], check=True)

# ==========================================================
# Evaluate Model
# ==========================================================

def evaluate_model(model_name):

    
    print(f"EVALUATING {model_name}")
    

    model_path = os.path.join(

        RESULTS_ROOT,

        model_name,

        "checkpoints",

        "best_model.pth"

    )

    output_dir = os.path.join(

        RESULTS_ROOT,

        model_name,

        "evaluation"

    )

    subprocess.run([

        sys.executable,

        "evaluate.py",

        "--model_name",

        model_name,

        "--data_dir",

        DATASET_PATH,

        "--model_path",

        model_path,

        "--output_dir",

        output_dir

    ], check=True)

# ==========================================================
# Benchmark Model
# ==========================================================

def benchmark_model(model_name):

    
    print(f"BENCHMARKING {model_name}")
    

    subprocess.run([

        sys.executable,

        "benchmark.py",

        "--model_name",

        model_name,

        "--output_dir",

        RESULTS_ROOT

    ], check=True)

# ==========================================================
# Aggregate Results
# ==========================================================

def aggregate_results():

    
    print("CREATING FINAL COMPARISON TABLE")
    

    subprocess.run([

        sys.executable,

        "aggregate_results.py",

        "--results_root",

        RESULTS_ROOT

    ], check=True)

# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    start_time = time.time()

   
    print("CERVICAL CANCER MODEL BENCHMARK PIPELINE")
    

    print(f"\nModels to train: {len(MODELS)}")

    for index, model_name in enumerate(MODELS, start=1):

        
        print(f"MODEL {index}/{len(MODELS)} : {model_name}")
        

        train_model(model_name)

        evaluate_model(model_name)

        benchmark_model(model_name)

    aggregate_results()

    total_time = (time.time() - start_time) / 60

    
    print("ALL EXPERIMENTS COMPLETED")
    print(f"\nTotal Runtime : {total_time:.2f} minutes")