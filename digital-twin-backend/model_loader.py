import os
import sys
import joblib
import torch
import functools

# Import real classes
import hybrid_model
from hybrid_model import HybridEnsemble, ImprovedTransformer


def load_hybrid_model():
    # 🔴 CRITICAL: Alias __main__ for pickle compatibility
    sys.modules["__main__"] = hybrid_model

    # Force CPU loading
    torch_load_original = torch.load
    torch.load = functools.partial(
        torch.load, map_location=torch.device("cpu")
    )

    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, "hybrid_model.pkl")

        model = joblib.load(model_path)

    finally:
        torch.load = torch_load_original

    return model


if __name__ == "__main__":
    model = load_hybrid_model()
    print("✅ Hybrid model loaded successfully")
    print("Input dim:", model.input_dim)
    print("Models:", model.models.keys())
