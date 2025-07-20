
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
import joblib
from pathlib import Path
from typing import List, Tuple, Optional
import sys

# ----------------------------- #
# 1. Core hyper-parameters       #
# ----------------------------- #
WIN_BEFORE   = 0.25      # seconds before peak
WIN_AFTER    = 0.45      # seconds after peak
VECTOR_LEN   = 40        # resample length L
CONTAM       = 0.05      # IsolationForest contamination
KEEP_PCT     = 40        # percentile cut-off for prominence
MIN_LEN_RAW  = 6        # discard windows with < this many raw samples
CHANNELS = [
    # --- original motion signals ---
    "acc_x", "acc_y", "acc_z",
    "lin_acc_x", "lin_acc_y", "lin_acc_z",
    # --- orientation (quats) ---
    "quat_w", "quat_x", "quat_y", "quat_z",
    # --- magnetometer ---
    "mag_x", "mag_y", "mag_z",
    # --- NEW derived features ---
    "ang_vel_x", "ang_vel_y", "ang_vel_z",
    "sin_roll", "cos_roll", "sin_pitch", "cos_pitch"
]



# ------------------------------------------------------- #
# ANGULAR-VELOCITY FROM QUATERNIONS                       #
# ≈ finite-difference:  ω = 2 * (q̇ ⊗ q⁻¹)_vector          #
# ------------------------------------------------------- #

def quat_normalize(Q):
    return Q / np.linalg.norm(Q, axis=1, keepdims=True)

def quaternion_to_angvel(quats: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    quats : (N, 4) w, x, y, z   ̶—̶  must be unit-norm
    t     : (N,)  seconds
    Returns
    -------
    ω : (N, 3)  rad/s  (x, y, z body axes)
    """
    q = quat_normalize(quats)
    dt = np.gradient(t)                    # uneven sampling OK
    dt[dt==0] = np.nan

    # quaternion derivative via central diff
    q_dot = np.gradient(q, axis=0) / dt[:, None]

    # replace NaNs with previous finite row
    nan_rows = np.any(np.isnan(q_dot), axis=1)
    q_dot[nan_rows] = np.nan_to_num(q_dot[nan_rows], nan=0.0)

    # conjugate of q
    q_conj = q * np.array([1, -1, -1, -1])

    # quaternion multiply q_dot ⊗ q_conj  (vectorised)
    w0, x0, y0, z0 = q_dot.T
    w1, x1, y1, z1 = q_conj.T

    mult_x =  w0*x1 + x0*w1 + y0*z1 - z0*y1
    mult_y =  w0*y1 - x0*z1 + y0*w1 + z0*x1
    mult_z =  w0*z1 + x0*y1 - y0*x1 + z0*w1

    omega = 2 * np.vstack([mult_x, mult_y, mult_z]).T
    return omega

# ------------------------------------------------------- #
# 2. Utility: resample a (n_samples, n_ch) matrix to L     #
# ------------------------------------------------------- #
def to_fixed_length(mat: np.ndarray, L: int = VECTOR_LEN) -> np.ndarray:
    n = mat.shape[0]
    x_old = np.linspace(0, 1, n)
    x_new = np.linspace(0, 1, L)
    cols = [np.interp(x_new, x_old, mat[:, i]) for i in range(mat.shape[1])]
    return np.stack(cols, axis=1).flatten()

# ------------------------------------------------------- #
# 3. Main training routine                                #
# ------------------------------------------------------- #
def run_inference(csv_path: str,
                  model_path: str,
                  channels: Optional[List[str]] = None,
                  keep_pct: int = KEEP_PCT,
                  verbose: bool = True) -> pd.DataFrame:
    """
    Train Isolation Forest on squash-swing IMU data.

    Parameters
    ----------
    csv_path  : path to raw BNO055 CSV
    keep_pct  : percentile of prominence threshold (lower → more peaks kept)
    contam    : expected fraction of anomalies for IsolationForest
    channels  : sensor columns to use; default = CHANNELS
    verbose   : print progress

    Returns
    -------
    sklearn.pipeline.Pipeline [StandardScaler → IsolationForest]
    """
    if channels is None:
        channels = CHANNELS

    # --- load & basic prep ---
    df = pd.read_csv(csv_path)
    df["t_s"] = (df["timestamp"] - df["timestamp"].iloc[0]) / 1000.0
    dt = np.median(np.diff(df["t_s"]))
    min_samples = int(0.6 / dt)          # ≈ 0.6 s separation
    df["lin_mag"] = np.linalg.norm(
        df[["lin_acc_x", "lin_acc_y", "lin_acc_z"]].values, axis=1)
    
    # sin / cos of roll & pitch  (degrees → radians)
    df["sin_roll"]  = np.sin(np.deg2rad(df["roll"]))
    df["cos_roll"]  = np.cos(np.deg2rad(df["roll"]))
    df["sin_pitch"] = np.sin(np.deg2rad(df["pitch"]))
    df["cos_pitch"] = np.cos(np.deg2rad(df["pitch"]))

    # 2. Angular velocity from quaternions
    quat_cols = ["quat_w", "quat_x", "quat_y", "quat_z"]
    omega = quaternion_to_angvel(df[quat_cols].values, df["t_s"].values)
    df[["ang_vel_x", "ang_vel_y", "ang_vel_z"]] = omega


    # --- wide-open peak scan ---
    peaks0, props0 = find_peaks(df["lin_mag"],
                                prominence=(None, None),
                                distance=min_samples)
    prom_all = props0["prominences"]
    peak_prom = np.percentile(prom_all, keep_pct)

    if verbose:
        print(f"[INFO] Found {len(peaks0)} bumps, "
              f"cut at {keep_pct}th pct → PEAK_PROM={peak_prom:.2f}")

    # --- final peaks with threshold ---
    peaks, _ = find_peaks(df["lin_mag"],
                          prominence=peak_prom,
                          distance=min_samples)
    if verbose:
        print(f"[INFO] Retained {len(peaks)} swing centres")

    # --- build windows ---
    windows = []
    for p in peaks:
        t0 = df.at[p, "t_s"]
        mask = (df["t_s"] >= t0 - WIN_BEFORE) & (df["t_s"] <= t0 + WIN_AFTER)
        w = df.loc[mask]
        if len(w) >= MIN_LEN_RAW:
            windows.append(w)

    if verbose:
        print(f"[INFO] Windows kept after length filter: {len(windows)}")

    if not windows:
        raise RuntimeError("No valid swing windows – adjust parameters.")

    # --- vectorise ---
    vectors = [to_fixed_length(w[channels].values) for w in windows]
    X_test = pd.DataFrame(vectors)

    # ---------- 4. load fitted pipeline & score
    try:
        pipe   = joblib.load(Path(model_path))
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        sys.exit(1)
        
    scores = pipe.score_samples(X_test)     # higher = more normal
    labels = pipe.predict(X_test)           #  1 = normal,  -1 = anomaly

    # ---------- 5. return nice DataFrame
    out = pd.DataFrame({
        "anomaly_score" : scores,
        "label"         : labels
    })
    if verbose:
        print(f"[INFO] Inference finished – {len(out)} swings scored")
    return out
