
import base64
import io
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from dataset import get_transforms
from model import get_model, get_input_size
from explainability import create_explainability_overlay


# --------------------------------------------------
# Configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = (
    PROJECT_ROOT
    / "benchmark_results"
    / "efficientnet_b0"
    / "checkpoints"
    / "best_model.pth"
)

MODEL_NAME = "efficientnet_b0"
CLASSES = ["CIN1", "CIN2", "CIN3", "Normal"]
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
GRADCAM_THRESHOLD = 0.4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = FastAPI(
    title="CerviScan Research API",
    description="Research prototype for cervical image classification.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# Load the existing trained model once
# --------------------------------------------------

model = None
transform = None
model_load_error = None


@app.on_event("startup")
def load_trained_model():
    global model, transform, model_load_error

    try:
        if not MODEL_PATH.is_file():
            raise FileNotFoundError(
                f"Model checkpoint not found: {MODEL_PATH}"
            )

        img_size = get_input_size(MODEL_NAME)
        _, transform = get_transforms(img_size=img_size)

        model = get_model(
            num_classes=len(CLASSES),
            model_name=MODEL_NAME,
            pretrained=False,
        )

        state_dict = torch.load(
            MODEL_PATH,
            map_location=DEVICE,
            weights_only=True,
        )

        model.load_state_dict(state_dict)
        model.to(DEVICE)
        model.eval()

        model_load_error = None
        print(f"Loaded trained model on {DEVICE}")

    except Exception as exc:
        model = None
        model_load_error = str(exc)
        print(f"Model loading failed: {exc}")


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def image_to_data_url(image_array, is_heatmap=False):
    """Convert an image array into a PNG data URL."""

    array = np.asarray(image_array)

    if is_heatmap:
        if array.dtype != np.uint8:
            array = np.clip(array * 255, 0, 255).astype(np.uint8)

        array = cv2.applyColorMap(array, cv2.COLORMAP_JET)
        array = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)

    elif array.dtype != np.uint8:
        if array.max(initial=0) <= 1:
            array = array * 255

        array = np.clip(array, 0, 255).astype(np.uint8)

    image = Image.fromarray(array).convert("RGB")

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")

    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def calculate_experimental_swede_score(
    predicted_class,
    heatmap,
    threshold=GRADCAM_THRESHOLD,
):
    """
    Reproduce the experimental heuristic from predict.py.

    Four component scores come from a predicted-class lookup.
    Lesion size is estimated from thresholded Grad-CAM coverage.

    This is not a validated clinical Swede Score.
    """

    heatmap = np.asarray(heatmap)

    lesion_pct = (
        float(np.sum(heatmap > threshold)) / heatmap.size * 100
        if heatmap.size
        else 0.0
    )

    if lesion_pct > 15:
        suggested_size = 2
    elif lesion_pct > 5:
        suggested_size = 1
    else:
        suggested_size = 0

    swede_lookup = {
        "Normal": (0, 0, 0, 0),
        "CIN1": (1, 0, 0, 1),
        "CIN2": (1, 1, 1, 1),
        "CIN3": (2, 2, 2, 2),
    }

    aceto, margin, vessels, iodine = swede_lookup.get(
        predicted_class,
        (0, 0, 0, 0),
    )

    total = aceto + margin + vessels + iodine + suggested_size

    if total >= 7:
        risk_tag = "HIGH"
        risk_level = (
            "High Risk (HSIL+): Urgent biopsy and treatment "
            "evaluation required."
        )
    elif total >= 5:
        risk_tag = "INTERMEDIATE"
        risk_level = (
            "Intermediate Risk (CIN2+): Colposcopic biopsy "
            "highly recommended."
        )
    else:
        risk_tag = "LOW"
        risk_level = (
            "Low Risk: Likely Normal or low-grade. "
            "Routine follow-up per protocol."
        )

    return {
        "acetowhiteness": aceto,
        "margins": margin,
        "vessels": vessels,
        "iodine": iodine,
        "lesion_size": suggested_size,
        "lesion_coverage_percent": round(lesion_pct, 2),
        "total": total,
        "maximum": 10,
        "risk_tag": risk_tag,
        "risk_level": risk_level,
        "method": (
            "Experimental heuristic: four criteria use a "
            "predicted-class lookup; lesion size uses Grad-CAM "
            "activation coverage."
        ),
    }


# --------------------------------------------------
# API endpoints
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "CerviScan Research API is running.",
        "model_loaded": model is not None,
        "device": str(DEVICE),
    }


@app.get("/health")
def health():
    return {
        "status": "ok" if model is not None else "model_unavailable",
        "model_name": MODEL_NAME,
        "device": str(DEVICE),
        "error": model_load_error,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None or transform is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model is unavailable: {model_load_error}",
        )

    if file.content_type not in (
        "image/jpeg",
        "image/png",
        "image/webp",
    ):
        raise HTTPException(
            status_code=400,
            detail="Please upload a JPEG, PNG, or WebP image.",
        )

    contents = await file.read(MAX_UPLOAD_BYTES + 1)

    if not contents:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty.",
        )

    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail="The image must be 10 MB or smaller.",
        )

    temp_path = None

    try:
        # Validate the actual image content.
        with Image.open(io.BytesIO(contents)) as uploaded_image:
            uploaded_image.verify()

        with Image.open(io.BytesIO(contents)) as uploaded_image:
            original_image = uploaded_image.convert("RGB")

        # Grad-CAM currently expects an image path.
        with tempfile.NamedTemporaryFile(
            suffix=".png",
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)
            original_image.save(temp_file, format="PNG")

        # Grad-CAM needs gradients, so do not wrap this call
        # in torch.no_grad() or torch.inference_mode().
        explanation = create_explainability_overlay(
            image_path=str(temp_path),
            model=model,
            transform=transform,
            model_name=MODEL_NAME,
            device=str(DEVICE),
            threshold=GRADCAM_THRESHOLD,
        )

        # Obtain class probabilities from the trained model.
        input_tensor = transform(original_image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = model(input_tensor)
            probabilities = (
                torch.softmax(logits, dim=1)[0].cpu().numpy()
            )

        predicted_index = int(np.argmax(probabilities))
        predicted_class = CLASSES[predicted_index]

        # --------------------------------------------------
        # NEW: Experimental Swede Score calculation
        # --------------------------------------------------

        swede_score = calculate_experimental_swede_score(
            predicted_class=predicted_class,
            heatmap=explanation["heatmap"],
            threshold=GRADCAM_THRESHOLD,
        )

        # Convert the existing visualization arrays to data URLs.
        images = {
            "original": image_to_data_url(np.array(original_image)),
            "heatmap": image_to_data_url(
                explanation["heatmap"],
                is_heatmap=True,
            ),
            "overlay": image_to_data_url(explanation["overlay"]),
            "annotated": image_to_data_url(explanation["annotated"]),
        }

        return {
            "predicted_class": predicted_class,
            "confidence": float(probabilities[predicted_index]),
            "probabilities": {
                class_name: float(probabilities[index])
                for index, class_name in enumerate(CLASSES)
            },

            # NEW: React can read these values from results.swede_score
            "swede_score": swede_score,

            "images": images,
            "bbox": explanation.get("bbox"),
            "disclaimer": (
                "Research prototype only. Predictions and Grad-CAM "
                "visualizations are not a clinical diagnosis. "
                "Model confidence is not a calibrated probability. "
                "The Swede Score shown is an experimental heuristic, "
                "not a validated clinical Swede Score."
            ),
        }

    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Unable to process this image: {exc}",
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {exc}",
        ) from exc

    finally:
        await file.close()

        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
