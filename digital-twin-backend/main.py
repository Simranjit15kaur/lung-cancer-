import numpy as np
import shap
import base64
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from io import BytesIO

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

from model_loader import load_hybrid_model
from predictor import predict_with_hybrid
from timeseries_builder import build_time_series
import torch

# ---------------- FastAPI ----------------
app = FastAPI(title="Digital Twin – Lung Cancer Detection", version="4.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- Load Hybrid Model ----------------
hybrid_model = load_hybrid_model()

# ---------------- FIXED SHAP MODEL WRAPPER ----------------
def shap_single_output(voc_batch, class_idx):
    """
    SHAP expects a SINGLE SCALAR output.
    So we return ONLY probability of class_idx.
    """

    ts_batch = []
    for row in voc_batch:
        ts = build_time_series(row).reshape(1620,)
        ts_batch.append(ts)
    ts_batch = np.array(ts_batch)

    # TREE MODELS
    xgb_probs = hybrid_model.models["xgboost"].predict_proba(ts_batch)
    lgb_probs = hybrid_model.models["lightgbm"].predict_proba(ts_batch)

    # TRANSFORMER
    transformer = hybrid_model.models["transformer"]
    transformer.eval()

    X_scaled = hybrid_model.scalers["transformer"].transform(ts_batch)

    with torch.no_grad():
        logits = transformer(torch.FloatTensor(X_scaled))
        trf_probs = torch.softmax(logits, dim=1).cpu().numpy()

    # Weighted ensemble
    final_probs = (
        0.3 * xgb_probs +
        0.4 * lgb_probs +
        0.3 * trf_probs
    )

    # RETURN ONLY ONE CLASS PROBABILITY
    return final_probs[:, class_idx]


# Background dataset for SHAP (10 samples × 27 features)
background = np.zeros((10, 27))


# ---------------- Pydantic Schemas ----------------
class SensorInput(BaseModel):
    features: List[float]


class ExplainRequest(BaseModel):
    features: List[float]
    voc_names: List[str]
    predicted_class: str


# ---------------- ROUTES ----------------
@app.get("/")
def root():
    return {"status": "Backend running"}


@app.post("/predict")
def predict(data: SensorInput):

    if len(data.features) != 27:
        return {"error": "Expected 27 VOC values"}

    ts_1620 = build_time_series(data.features)  # (1,1620)

    result = predict_with_hybrid(hybrid_model, ts_1620)

    return result


@app.post("/explain")
def explain(data: ExplainRequest):

    x = np.array(data.features).reshape(1, 27)

    class_map = {"Benign": 0, "Cancer": 1, "Control": 2}
    idx = class_map[data.predicted_class]

    # Create a class-specific SHAP explainer
    class_explainer = shap.KernelExplainer(
        lambda v: shap_single_output(v, idx),
        background
    )

    shap_vals = class_explainer.shap_values(x)[0]   # returns 27 SHAP values

    # Select top 6 VOCs
    abs_vals = np.abs(shap_vals)
    top_idx = np.argsort(abs_vals)[::-1][:6]
    top_names = np.array(data.voc_names)[top_idx]
    top_vals = shap_vals[top_idx]

    # Plot bar chart
    plt.figure(figsize=(8, 4))
    colors = ["red" if v > 0 else "blue" for v in top_vals]
    plt.barh(top_names, top_vals, color=colors)
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=150)
    buf.seek(0)
    img64 = base64.b64encode(buf.read()).decode()
    plt.close()

    return {
        "top_voc_names": top_names.tolist(),
        "top_voc_values": top_vals.tolist(),
        "bar_plot": img64
    }
