from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import numpy as np
import joblib
from xgboost import XGBClassifier
import tensorflow as tf
from scipy.stats import skew, kurtosis
from scipy.fft import rfft

# ---------------- FastAPI + CORS ----------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------- Request body from frontend -------------
class SensorData(BaseModel):
    vibration_x: float
    vibration_y: float
    vibration_z: float
    acoustic_level: float
    temperature: float


# ------------- Load trained models + scaler + encoder -------------
MODEL_DIR = "models"

scaler_feats = joblib.load(f"{MODEL_DIR}/scaler_features.pkl")
label_encoder = joblib.load(f"{MODEL_DIR}/label_encoder.pkl")

mlp_model = tf.keras.models.load_model(f"{MODEL_DIR}/mlp_feats.h5")

xgb_model = XGBClassifier()
xgb_model.load_model(f"{MODEL_DIR}/xgb_feats.json")

# these must match notebook
TIMESTEPS = 20
N_BANDS = 6
N_CHANNELS = 5  # Vx, Vy, Vz, Acoustic, Temp


# ------------- Feature engineering (EXACT same as notebook) -------------
def time_feats(window: np.ndarray) -> np.ndarray:
    """
    Same as time_feats(window) in your Colab:
    mean, std(ddof=0), RMS, P2P, IQR, skew, kurtosis, crest-ish, mean diff
    for each of the 5 channels.
    """
    feats = []
    for ch in range(window.shape[1]):
        arr = window[:, ch]
        feats += [
            arr.mean(),                             # 1 mean
            arr.std(ddof=0),                        # 2 std
            np.sqrt(np.mean(arr**2)),               # 3 RMS
            np.max(arr) - np.min(arr),              # 4 P2P
            np.percentile(arr, 75) - np.percentile(arr, 25),  # 5 IQR
            skew(arr),                              # 6 skew
            kurtosis(arr),                          # 7 kurtosis
            (np.max(np.abs(arr)) /
             (np.mean(np.abs(arr)) + 1e-9)),        # 8 crest-ish
            np.mean(np.diff(arr)),                  # 9 mean diff
        ]
    return np.array(feats)


def freq_feats(window: np.ndarray, n_bands: int = N_BANDS) -> np.ndarray:
    """
    Same as freq_feats(window, n_bands=6) in notebook:
    FFT band energy normalized for each channel.
    """
    feats = []
    t = window.shape[0]
    fft_idx_edges = np.linspace(0, t // 2 + 1, n_bands + 1, dtype=int)

    for ch in range(window.shape[1]):
        sig = window[:, ch]
        fft_vals = np.abs(rfft(sig))
        energy = (fft_vals ** 2).sum() + 1e-9

        for b in range(n_bands):
            v = fft_vals[fft_idx_edges[b]:fft_idx_edges[b + 1]]
            feats.append((v ** 2).sum() / energy)

    return np.array(feats)


def build_feature_vector_from_raw(raw_row: np.ndarray) -> np.ndarray:
    """
    Same logic as predict_from_raw_row_safe in Colab:
    - tile raw_row into TIMESTEPS window
    - compute time_feats + freq_feats
    - handle NaNs
    - return (1, 75) feature vector
    """
    # window shape: (TIMESTEPS, 5)
    w = np.tile(raw_row.reshape(1, -1), (TIMESTEPS, 1))

    t_feats = time_feats(w)
    f_feats = freq_feats(w, n_bands=N_BANDS)
    feats = np.concatenate([t_feats, f_feats])  # length 75

    # avoid NaNs from skew/kurtosis on constant signals
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

    return feats.reshape(1, -1)  # (1, 75)


# -------------------- Routes --------------------
@app.get("/")
def home():
    return {"message": "Rotating Equipment Prediction API (XGB + MLP ensemble)"}


@app.post("/predict")
def predict(data: SensorData):
    # 1. raw sensor row in SAME ORDER as notebook & frontend
    raw = np.array(
        [
            data.vibration_x,
            data.vibration_y,
            data.vibration_z,
            data.acoustic_level,
            data.temperature,
        ],
        dtype=float,
    )

    # 2. feature vector (1, 75)
    feat_vec = build_feature_vector_from_raw(raw)

    # 3. scale
    feat_scaled = scaler_feats.transform(feat_vec)

    # 4. model probabilities
    xgb_proba = xgb_model.predict_proba(feat_scaled)[0]
    mlp_proba = mlp_model.predict(feat_scaled)[0]

    # 5. ensemble (soft vote)
    avg_proba = (xgb_proba + mlp_proba) / 2.0
    idx = int(np.argmax(avg_proba))

    fault_label = label_encoder.inverse_transform([idx])[0]
    probability = float(avg_proba[idx])

    return {
        "fault_label": fault_label,
        "fault_code": idx,
        "probability": round(probability, 4),
    }
