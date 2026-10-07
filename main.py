"""
Floret Scans — FastAPI backend.

Serves a /predict endpoint: send an image, get back the predicted flower
species, confidence score, and description.

Run:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000

Requires floret_scans_model.pth (from train.py) to be in the same folder.
"""

import io
import json

import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from torchvision import transforms, models
import torch.nn as nn

from labels import CLASS_NAMES

app = FastAPI(title="Floret Scans API")

# Allow the React frontend (running on a different port) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this to your frontend's actual URL before deploying
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONFIDENCE_THRESHOLD = 0.4  # below this, we say "not confident" instead of guessing

# --- Load model once at startup ---
def load_model():
    model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, len(CLASS_NAMES))
    model.load_state_dict(torch.load("floret_scans_model.pth", map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

model = load_model()

# --- Load descriptions once at startup ---
with open("descriptions.json") as f:
    DESCRIPTIONS = json.load(f)

# Same preprocessing used for validation in train.py
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


@app.get("/")
def health_check():
    return {"status": "ok", "message": "Floret Scans API is running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    image_bytes = await file.read()
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image file")

    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)

    confidence = confidence.item()
    predicted_idx = predicted_idx.item()
    species_name = CLASS_NAMES[predicted_idx]

    if confidence < CONFIDENCE_THRESHOLD:
        return {
            "confident": False,
            "message": "Not confident this is a flower, or couldn't identify the species clearly.",
            "best_guess": species_name,
            "confidence": round(confidence, 3),
        }

    info = DESCRIPTIONS.get(species_name, {
        "family": "unknown",
        "description": "No description available yet for this species.",
    })

    return {
        "confident": True,
        "species": species_name,
        "confidence": round(confidence, 3),
        "family": info["family"],
        "description": info["description"],
    }