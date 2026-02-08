import numpy as np
import torch

LABELS = ["Benign", "Cancer", "Control"]

def predict_with_hybrid(model, X):
    """
    model : loaded HybridEnsemble
    X     : np.array of shape (1, 1620)
    """

    # ---------- Scale for transformer ----------
    X_scaled = model.scalers["transformer"].transform(X)

    # ---------- XGBoost ----------
    xgb_probs = model.models["xgboost"].predict_proba(X)

    # ---------- LightGBM ----------
    lgb_probs = model.models["lightgbm"].predict_proba(X)

    # ---------- Transformer ----------
    transformer = model.models["transformer"]
    transformer.eval()

    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_scaled)
        logits = transformer(X_tensor)
        transformer_probs = torch.softmax(logits, dim=1).cpu().numpy()

    # ---------- Weighted Ensemble ----------
    weights = [0.3, 0.4, 0.3]
    final_probs = (
        weights[0] * xgb_probs +
        weights[1] * lgb_probs +
        weights[2] * transformer_probs
    )

    pred_idx = int(np.argmax(final_probs))
    confidence = float(final_probs[0, pred_idx])

    return {
        "prediction": LABELS[pred_idx],
        "confidence": round(confidence, 4),
        "probabilities": {
            LABELS[i]: round(float(final_probs[0, i]), 4)
            for i in range(len(LABELS))
        }
    }
