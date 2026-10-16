from pathlib import Path

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "Split_Data_PatientWise"

OUTPUT_ROOT = PROJECT_ROOT / "outputs_merged_patientwise_20261013"

BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 5
SEED = 42


# MODELS TO BENCHMARK


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