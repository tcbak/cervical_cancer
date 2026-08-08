import torch
import torch.nn as nn
import torchvision.models as models

# ==========================================================
# Model Metadata
# ==========================================================
# All benchmark models and their required input resolutions.
# This dictionary serves as the single source of truth for the
# benchmarking framework.

MODEL_INFO = {
    "efficientnet_b0": {
        "display_name": "EfficientNet-B0",
        "category": "Proposed",
        "input_size": (224, 224)
    },

    "mobilenet_v2": {
        "display_name": "MobileNetV2",
        "category": "Lightweight",
        "input_size": (224, 224)
    },

    "shufflenet_v2_x1_0": {
        "display_name": "ShuffleNetV2 x1.0",
        "category": "Ultra-Lightweight",
        "input_size": (224, 224)
    },

    "squeezenet1_1": {
        "display_name": "SqueezeNet1.1",
        "category": "Ultra-Lightweight",
        "input_size": (224, 224)
    },

    "densenet121": {
        "display_name": "DenseNet121",
        "category": "Medium",
        "input_size": (224, 224)
    },

    "resnet18": {
        "display_name": "ResNet18",
        "category": "Medium",
        "input_size": (224, 224)
    },

    "resnet34": {
        "display_name": "ResNet34",
        "category": "Medium",
        "input_size": (224, 224)
    },

    "resnet50": {
        "display_name": "ResNet50",
        "category": "Heavy",
        "input_size": (224, 224)
    },

    # Optional architectures (kept for compatibility)
    "vgg16": {
        "display_name": "VGG16",
        "category": "Heavy",
        "input_size": (224, 224)
    },

    "inception_v3": {
        "display_name": "InceptionV3",
        "category": "Heavy",
        "input_size": (299, 299)
    }
}


# ==========================================================
# Helper Functions
# ==========================================================

def list_supported_models():
    """
    Returns a list of all supported model names.
    """
    return list(MODEL_INFO.keys())


def get_model_info(model_name):
    """
    Returns metadata for a given model.
    """
    if model_name not in MODEL_INFO:
        raise ValueError(
            f"Unsupported model '{model_name}'. "
            f"Supported models: {list_supported_models()}"
        )

    return MODEL_INFO[model_name]


def get_input_size(model_name):
    """
    Returns the required input resolution for a model.
    """
    return get_model_info(model_name)["input_size"]


def _set_trainable(model, freeze_backbone):
    """
    Freeze or unfreeze all model parameters.
    """
    for param in model.parameters():
        param.requires_grad = not freeze_backbone


# ==========================================================
# Model Factory
# ==========================================================

def get_model(
        num_classes=4,
        model_name="efficientnet_b0",
        pretrained=True,
        freeze_backbone=False):
    """
    Builds and returns a CNN backbone.

    Supported architectures
    -----------------------
    efficientnet_b0
    mobilenet_v2
    shufflenet_v2_x1_0
    squeezenet1_1
    densenet121
    resnet18
    resnet34
    resnet50
    vgg16
    inception_v3
    """

    print(
        f"Initializing {model_name} "
        f"(pretrained={pretrained}, "
        f"freeze_backbone={freeze_backbone})..."
    )

    weights = "DEFAULT" if pretrained else None

        # ======================================================
    # EfficientNet-B0
    # ======================================================
    if model_name == "efficientnet_b0":

        model = models.efficientnet_b0(weights=weights)

        _set_trainable(model, freeze_backbone)

        in_features = model.classifier[1].in_features

        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, num_classes)
        )

    # ======================================================
    # MobileNetV2
    # ======================================================
    elif model_name == "mobilenet_v2":

        model = models.mobilenet_v2(weights=weights)

        _set_trainable(model, freeze_backbone)

        in_features = model.classifier[1].in_features

        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features, num_classes)
        )

    # ======================================================
    # ShuffleNetV2
    # ======================================================
    elif model_name == "shufflenet_v2_x1_0":

        model = models.shufflenet_v2_x1_0(weights=weights)

        _set_trainable(model, freeze_backbone)

        in_features = model.fc.in_features

        model.fc = nn.Linear(
            in_features,
            num_classes
        )

    # ======================================================
    # SqueezeNet1.1
    # ======================================================
    elif model_name == "squeezenet1_1":

        model = models.squeezenet1_1(weights=weights)

        _set_trainable(model, freeze_backbone)

        # Replace classifier
        model.classifier[1] = nn.Conv2d(
            in_channels=512,
            out_channels=num_classes,
            kernel_size=1
        )

        model.num_classes = num_classes

    # ======================================================
    # DenseNet121
    # ======================================================
    elif model_name == "densenet121":

        model = models.densenet121(weights=weights)

        _set_trainable(model, freeze_backbone)

        in_features = model.classifier.in_features

        model.classifier = nn.Linear(
            in_features,
            num_classes
        )

    # ======================================================
    # ResNet18
    # ======================================================
    elif model_name == "resnet18":

        model = models.resnet18(weights=weights)

        _set_trainable(model, freeze_backbone)

        in_features = model.fc.in_features

        model.fc = nn.Linear(
            in_features,
            num_classes
        )

    # ======================================================
    # ResNet34
    # ======================================================
    elif model_name == "resnet34":

        model = models.resnet34(weights=weights)

        _set_trainable(model, freeze_backbone)

        in_features = model.fc.in_features

        model.fc = nn.Linear(
            in_features,
            num_classes
        )

            # ======================================================
    # ResNet50
    # ======================================================
    elif model_name == "resnet50":

        model = models.resnet50(weights=weights)

        _set_trainable(model, freeze_backbone)

        in_features = model.fc.in_features

        model.fc = nn.Linear(
            in_features,
            num_classes
        )

    # ======================================================
    # VGG16 (Optional)
    # ======================================================
    elif model_name == "vgg16":

        model = models.vgg16(weights=weights)

        _set_trainable(model, freeze_backbone)

        in_features = model.classifier[6].in_features

        model.classifier[6] = nn.Linear(
            in_features,
            num_classes
        )

    # ======================================================
    # Inception V3 (Optional)
    # ======================================================
    elif model_name == "inception_v3":

        model = models.inception_v3(
            weights=weights,
            aux_logits=True,
            init_weights=False
        )

        _set_trainable(model, freeze_backbone)

        in_features = model.fc.in_features

        model.fc = nn.Linear(
            in_features,
            num_classes
        )

        # Disable auxiliary classifier for consistent outputs
        model.aux_logits = False
        model.AuxLogits = None

    else:

        raise ValueError(
            f"Unsupported model '{model_name}'.\n"
            f"Supported models:\n{list_supported_models()}"
        )

    return model


# ==========================================================
# Standalone Test
# ==========================================================

if __name__ == "__main__":

    print("=" * 80)
    print("Testing Supported Models")
    print("=" * 80)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"\nRunning on: {device}\n")

    for model_name in list_supported_models():

        print("-" * 80)

        try:

            info = get_model_info(model_name)

            model = get_model(
                num_classes=4,
                model_name=model_name,
                pretrained=False
            )

            model = model.to(device)

            h, w = get_input_size(model_name)

            dummy = torch.randn(
                2,
                3,
                h,
                w
            ).to(device)

            output = model(dummy)

            total_params = sum(
                p.numel()
                for p in model.parameters()
            )

            trainable_params = sum(
                p.numel()
                for p in model.parameters()
                if p.requires_grad
            )

            print(f"Model          : {info['display_name']}")
            print(f"Category       : {info['category']}")
            print(f"Input Size     : {h} x {w}")
            print(f"Parameters     : {total_params:,}")
            print(f"Trainable      : {trainable_params:,}")
            print(f"Output Shape   : {tuple(output.shape)}")
            print("Status         : PASS")

        except Exception as e:

            print(f"Model          : {model_name}")
            print("Status         : FAILED")
            print(e)

    
    print("Model Testing Completed")
    