from pathlib import Path

# ==========================================================
# PROJECT PATHS
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "Split_Data"

OUTPUT_ROOT = PROJECT_ROOT / "outputs_benchmark"

# ==========================================================
# DATASET
# ==========================================================

NUM_CLASSES = 4

CLASS_NAMES = [
    "CIN1",
    "CIN2",
    "CIN3",
    "Normal"
]

# ==========================================================
# TRAINING SETTINGS
# ==========================================================

BATCH_SIZE = 32

EPOCHS = 20

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4

EARLY_STOPPING_PATIENCE = 5

# ==========================================================
# RANDOM SEED
# ==========================================================

RANDOM_SEED = 42

# ==========================================================
# MODELS TO BENCHMARK
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