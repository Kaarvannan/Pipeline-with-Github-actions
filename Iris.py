from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from huggingface_hub import hf_hub_download
import os
import importlib
from huggingface_hub import login
from huggingface_hub import hf_hub_download
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST

from fastapi import Response

REQUEST_COUNT = Counter("iris_requests_total", "Total prediction requests")


app = FastAPI(title="Iris Classifier API")

Instrumentator().instrument(app).expose(app)

# --- Step 1: Download model from Hugging Face ---
try:
    model_path = hf_hub_download(
        repo_id="Kaar7/Iris",
        filename="model.pkl",
        token=os.getenv("HF_TOKEN")
)
except Exception as e:
    raise RuntimeError(f"Failed to download model from Hugging Face: {e}")

# --- Step 2: Verify file exists ---
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")

# --- Step 3: Load model safely ---
try:
    model = joblib.load(model_path)
except Exception as e:
    # Handle scikit-learn version mismatch gracefully
    import sklearn
    version = sklearn.__version__
    raise RuntimeError(
        f"Error loading model.pkl with sklearn {version}. "
        f"This model was likely trained with a different version. "
        f"Try installing sklearn==1.2.2\nFull error: {e}"
    )

# --- Step 4: Input schema ---
class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# --- Step 5: Routes ---


@app.get("/")
def home():
    return {"message": "✅ Iris prediction API is running!"}


@app.post("/predict")
def predict(features: IrisFeatures):
    data = np.array([[features.sepal_length, features.sepal_width,
                      features.petal_length, features.petal_width]])
    prediction = model.predict(data)
    return {"prediction": int(prediction[0])}
