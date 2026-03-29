import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import matplotlib.pyplot as plt

# =========================================================
# PATH SETUP
# =========================================================
HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[1]          # experiments -> src -> core
PROJECT_ROOT = REPO_ROOT.parent      # Music_AI_1.1

sys.path.append(str(REPO_ROOT))
sys.path.append(str(HERE))

# ---------------------------------------------------------
# IMPORT FROM YOUR EXISTING AUDIO SCRIPT
# ---------------------------------------------------------
from src.models.train_instrument import AudioSynthTrainer
from src.models.signal_processing import harmonic_synthesis, noise_synthesis

from src.curve.curve_audio_core import (
    AnalysisMeta,
    ensure_dir,
    maybe_round_df,
    moving_average,
    median_filter_1d,
    normalize_01,
    hz_to_midi,
    midi_to_hz,
    load_audio_mono,
    load_audio_model,
    resize_control_width,
    resample_2d_rows,
    sort_feature_cols,
    enforce_dense_compact_layout,
    compute_framewise_attributes,
    estimate_compact_controls_from_audio,
    build_formula_based_controls,
    predict_synth_controls_from_pitch_loudness,
    build_compact_dataframe_from_controls,
    reconstruct_from_compact_csv,
)

# ---------------------------------------------------------
# IMPORT FROM YOUR IMAGE/CURVE SCRIPT
# ---------------------------------------------------------
from src.curve.curve_utils import run_pipeline

# =========================================================
# USER CONFIG
# =========================================================
RUN_PREFIX = "curve_image_ratio_prototype"

IMAGE_PATH = Path(r"C:\curve_simple\input\img1.png")
OUTPUT_DIR = Path(r"C:\curve_simple\output\curve_compact_generation_output9")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_WAV = Path(r"C:\curve_simple\input\Input.wav")
CKPT_PATH = Path(r"C:\curve_simple\checkpoints\Guitar.ckpt")

GENERATION_MODE = "all"
# choices:
#   "formula_based"
#   "ml_based"
#   "audio_based"
#   "all"

ROUND_FLOATS_FOR_CSV = False
CSV_FLOAT_DECIMALS = 6

# =========================================================
# AUDIO CONFIG
# =========================================================
SR = 16000
N_FFT = 2048
HOP_LENGTH = 64
WIN_LENGTH = 2048
WINDOW = "hann"
FRAME_RATE = 250
EPS = 1e-8

YIN_FMIN = 40.0
YIN_FMAX = 2000.0
LOUDNESS_GATE_THRESHOLD = 0.01

NUM_HARMONICS = 100
NUM_NOISE_BANDS = 64

DEFAULT_NOISE_SCALE = 0.03
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================================
# CURVE EXTRACTION CONFIG
# =========================================================
ZERO_MODE = "horizontal"
ZERO_LINE_Y = 1000
ZERO_P1 = (0.0, 240.0)
ZERO_P2 = (1000.0, 240.0)

PREPROCESS_MODE = "canny"
GAUSSIAN_KERNEL = 5
CANNY_LOW = 50
CANNY_HIGH = 150

BOUNDARY_MODE = "upper"
USE_SKELETON = False
MIN_CURVE_POINTS = 30
MIN_RUN = 1
SPIKE_MAX_JUMP = 20.0

SMOOTH_WINDOW = 25
SMOOTH_METHOD = "savgol"
SMOOTH_POLYORDER = 3
SIMPLIFY_STEP = 10
PEAK_PROMINENCE = 8.0

# =========================================================
# CURVE -> CONTROL STRATEGY
# =========================================================
CURVE_PITCH_MIN_HZ = 70.0
CURVE_PITCH_MAX_HZ = 360.0
DONOR_PITCH_WEIGHT = 0.00
CURVE_PITCH_WEIGHT = 1.00

HEIGHT_LOUDNESS_RATIO_MIN = 0.45
HEIGHT_LOUDNESS_RATIO_MAX = 1.30
LOUDNESS_BLEND = 1.00

USE_EVENT_ATTACK_BOOST = True
EVENT_ATTACK_BOOST = 0.12
EVENT_ATTACK_WIDTH_FRAMES = 8

# turn this OFF for jumpier note transitions
APPLY_CURVE_PITCH_SMOOTHING = False
CURVE_PITCH_SMOOTH_WIN = 1

# ---------------------------------------------------------
# PEAK-FAITHFUL CURVE MAPPING
# ---------------------------------------------------------
CURVE_BASELINE_PERCENTILE = 18.0
CURVE_BASELINE_POWER = 1.20

PEAK_PRESERVE_WINDOW = 7
PEAK_PRESERVE_BLEND = 0.05

USE_PEAK_CAP = True
PEAK_CAP_PERCENTILE = 99.6

SOURCE_MEDIAN_WIN = 17
SOURCE_SPIKE_THRESHOLD = 2.5
SOURCE_SPIKE_BLEND = 0.15

SUPPRESS_NARROW_UPWARD_SPIKES = True
UPWARD_SPIKE_MEDIAN_WIN = 31
UPWARD_SPIKE_THRESHOLD_PX = 140.0
UPWARD_SPIKE_MAX_WIDTH = 10
UPWARD_SPIKE_REPAIR_BLEND = 0.10

RIGHT_TAIL_FRACTION = 0.12
RIGHT_TAIL_SPIKE_RATIO = 1.60
RIGHT_TAIL_BLEND = 0.10

# ---------------------------------------------------------
# TRUE RECTANGULAR NOTE-BLOCK PITCH
# ---------------------------------------------------------
USE_TRUE_NOTE_BLOCKS = True

# pooling before segmentation
NOTE_BLOCK_POOL_FRAMES = 24

# how much change required to start a new note block
NOTE_BLOCK_CHANGE_THRESHOLD_SEMITONES = 1.60

# minimum pooled-run length
NOTE_BLOCK_MIN_RUN = 2

# merge adjacent blocks if their centers are close
NOTE_BLOCK_MERGE_THRESHOLD_SEMITONES = 0.90

# expand as perfectly rectangular notes
NOTE_BLOCK_EDGE_RAMP_FRAMES = 0

# musical mapping
USE_SCALE_QUANTIZATION = True
QUANTIZER_MODE = "minor_pentatonic"
QUANTIZER_ROOT_PC = 0
QUANTIZE_BLOCK_CENTER_BLEND = 1.00

# if you want absolutely flat tops, keep both strict
BLOCK_ALLOW_WITHIN_NOTE_DEVIATION = 0.00
BLOCK_MICRO_WIGGLE_BLEND = 0.00

# optional donor influence OFF for clearer melody change
USE_DONOR_NOTE_GUIDE = False
DONOR_NOTE_GUIDE_BLEND = 0.00

# ---------------------------------------------------------
# TRANSITION CLEANUP (NEW)
# ---------------------------------------------------------
# nearest fill avoids creating linear ramps like 60 -> 110 -> 120 -> 166
USE_NEAREST_PITCH_FILL = True

# collapse short transition runs between stable notes
USE_TRANSITION_COLLAPSE = True
TRANSITION_MIN_RUN_FRAMES = 8
TRANSITION_JUMP_THRESHOLD_SEMITONES = 1.10

# if a short transition is very ambiguous, optionally zero it
TRANSITION_ZERO_AMBIGUOUS = False
TRANSITION_ZERO_MAX_RUN_FRAMES = 3

# =========================================================
# LOUDNESS SHAPING
# =========================================================
USE_MELODIC_LOUDNESS = True
LOUDNESS_HEIGHT_WEIGHT = 0.35
LOUDNESS_DONOR_WEIGHT = 0.65
LOUDNESS_DRIVER_SMOOTH_WIN = 21

USE_ATTACK_RELEASE_LOUDNESS = True
LOUDNESS_ATTACK_COEF = 0.45
LOUDNESS_RELEASE_COEF = 0.10

USE_PHRASE_GATE = True
PHRASE_GATE_ON = 0.055
PHRASE_GATE_OFF = 0.035

# ---------------------------------------------------------
# BANDED PITCH -> LOUDNESS MODEL
# ---------------------------------------------------------
USE_BANDED_PITCH_LOUDNESS = True
BANDED_LOUDNESS_BLEND = 0.60
BANDED_LOUDNESS_USE_MIDI = True
BANDED_LOUDNESS_MIDI_SPLITS = [52.0, 64.0]
BANDED_LOUDNESS_MIN_SAMPLES_PER_BAND = 24

# =========================================================
# AUDIO-BASED BLEND CONFIG
# =========================================================
AUDIO_BASED_USE_DIRECT_AMPLITUDE = False
AUDIO_BASED_USE_DIRECT_HARMONICS = True
AUDIO_BASED_USE_DIRECT_NOISE = True
AUDIO_BASED_MODULATE_AMPLITUDE_BY_INPUT = True

AMPLITUDE_SMOOTH_WIN = 7

AUDIO_BASED_DONOR_HARMONIC_BLEND = 0.72
AUDIO_BASED_APPLY_CURVE_BRIGHTNESS_TILT = True
AUDIO_BASED_BRIGHTNESS_TILT_STRENGTH = 1.10
AUDIO_BASED_DONOR_NOISE_BLEND = 0.75

# =========================================================
# OUTPUT FILES
# =========================================================
OUT_RAW_CURVE_CSV = OUTPUT_DIR / f"{RUN_PREFIX}_curve_mapping_raw.csv"
OUT_SMOOTH_CURVE_CSV = OUTPUT_DIR / f"{RUN_PREFIX}_curve_mapping_smooth.csv"
OUT_SIMPLIFIED_CURVE_CSV = OUTPUT_DIR / f"{RUN_PREFIX}_curve_mapping_simplified.csv"
OUT_EVENTS_CSV = OUTPUT_DIR / f"{RUN_PREFIX}_peaks_dips.csv"

OUT_HEIGHT_RAW_CSV = OUTPUT_DIR / "height_profile_raw.csv"
OUT_HEIGHT_SMOOTH_CSV = OUTPUT_DIR / "height_profile_smooth.csv"
OUT_HEIGHT_SIMPLIFIED_CSV = OUTPUT_DIR / "height_profile_simplified.csv"
OUT_HEIGHT_EVENTS_CSV = OUTPUT_DIR / "height_peaks_dips.csv"

OUT_RATIO_CSV = OUTPUT_DIR / f"{RUN_PREFIX}_curve_ratio_controls.csv"
OUT_PITCH_LOUDNESS_CSV = OUTPUT_DIR / f"{RUN_PREFIX}_pitch_loudness_bundle.csv"

OUT_AUDIO_REFERENCE_COMPACT_CSV = OUTPUT_DIR / f"{RUN_PREFIX}_audio_reference_compact_bundle.csv"
OUT_AUDIO_REFERENCE_FULL_META_JSON = OUTPUT_DIR / f"{RUN_PREFIX}_audio_reference_meta.json"

OUT_SUMMARY_PLOT = OUTPUT_DIR / f"{RUN_PREFIX}_summary_plot.png"
OUT_COMPONENTS_PLOT = OUTPUT_DIR / f"{RUN_PREFIX}_components_plot.png"
OUT_META_JSON = OUTPUT_DIR / f"{RUN_PREFIX}_meta.json"
OUT_LONG_VIS_CSV = OUTPUT_DIR / f"{RUN_PREFIX}_components_long.csv"

OUT_CLEANED_CURVE_CSV = OUTPUT_DIR / f"{RUN_PREFIX}_cleaned_curve_for_pitch.csv"

# =========================================================
# LOCAL HELPERS
# =========================================================
def resample_1d(x: np.ndarray, n_target: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if len(x) == n_target:
        return x.copy()
    if len(x) == 0:
        return np.zeros(n_target, dtype=np.float32)
    src = np.linspace(0.0, 1.0, len(x), dtype=np.float32)
    dst = np.linspace(0.0, 1.0, n_target, dtype=np.float32)
    return np.interp(dst, src, x).astype(np.float32)


def max_filter_1d(x: np.ndarray, window: int = 3) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if window <= 1 or len(x) == 0:
        return x.copy()

    r = window // 2
    out = np.empty_like(x)
    n = len(x)

    for i in range(n):
        left = max(0, i - r)
        right = min(n, i + r + 1)
        out[i] = np.max(x[left:right])

    return out.astype(np.float32)


def attack_release_filter(x: np.ndarray, attack: float, release: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if len(x) == 0:
        return x.copy()

    out = np.zeros_like(x, dtype=np.float32)
    out[0] = x[0]

    for i in range(1, len(x)):
        coef = attack if x[i] > out[i - 1] else release
        coef = float(np.clip(coef, 0.0, 1.0))
        out[i] = out[i - 1] + coef * (x[i] - out[i - 1])

    return out.astype(np.float32)


def fill_unvoiced_pitch_nearest(pitch_hz: np.ndarray) -> np.ndarray:
    x = np.asarray(pitch_hz, dtype=np.float32).copy()
    n = len(x)
    if n == 0:
        return x

    voiced = np.isfinite(x) & (x > 0)
    if np.sum(voiced) == 0:
        return np.full_like(x, 220.0, dtype=np.float32)

    voiced_idx = np.where(voiced)[0]
    unvoiced_idx = np.where(~voiced)[0]

    if len(voiced_idx) == 1:
        x[~voiced] = x[voiced_idx[0]]
        return x.astype(np.float32)

    for i in unvoiced_idx:
        nearest = voiced_idx[np.argmin(np.abs(voiced_idx - i))]
        x[i] = x[nearest]

    return x.astype(np.float32)


def fill_unvoiced_pitch_linear(pitch_hz: np.ndarray) -> np.ndarray:
    x = np.asarray(pitch_hz, dtype=np.float32).copy()
    idx = np.arange(len(x))

    voiced = np.isfinite(x) & (x > 0)
    if np.sum(voiced) == 0:
        return np.full_like(x, 220.0, dtype=np.float32)

    if np.sum(voiced) == 1:
        x[~voiced] = x[voiced][0]
        return x.astype(np.float32)

    x[~voiced] = np.interp(idx[~voiced], idx[voiced], x[voiced]).astype(np.float32)
    return x.astype(np.float32)


def collapse_short_pitch_transitions(
    pitch_hz: np.ndarray,
    min_run_frames: int = 8,
    jump_threshold_semitones: float = 1.10,
    zero_ambiguous: bool = False,
    zero_max_run_frames: int = 3,
) -> np.ndarray:
    x = np.asarray(pitch_hz, dtype=np.float32).copy()
    if len(x) == 0:
        return x

    valid = np.isfinite(x) & (x > 0)
    if np.sum(valid) < 2:
        return x

    midi = np.zeros_like(x, dtype=np.float32)
    midi[valid] = hz_to_midi(np.clip(x[valid], EPS, None)).astype(np.float32)

    delta = np.abs(np.diff(midi))
    boundaries = [0] + list(np.where(delta >= jump_threshold_semitones)[0] + 1) + [len(x)]

    runs: List[Tuple[int, int]] = []
    for i in range(len(boundaries) - 1):
        s = boundaries[i]
        e = boundaries[i + 1]
        if e > s:
            runs.append((s, e))

    if len(runs) < 3:
        return x

    out = x.copy()

    for i in range(1, len(runs) - 1):
        s, e = runs[i]
        run_len = e - s
        if run_len >= min_run_frames:
            continue

        ps, pe = runs[i - 1]
        ns, ne = runs[i + 1]

        prev_block = out[ps:pe]
        curr_block = out[s:e]
        next_block = out[ns:ne]

        if len(prev_block) == 0 or len(curr_block) == 0 or len(next_block) == 0:
            continue

        prev_val = float(np.median(prev_block))
        curr_val = float(np.median(curr_block))
        next_val = float(np.median(next_block))

        d_prev = abs(curr_val - prev_val)
        d_next = abs(curr_val - next_val)

        if zero_ambiguous and run_len <= zero_max_run_frames and abs(d_prev - d_next) < 0.10 * max(prev_val, EPS):
            out[s:e] = 0.0
        else:
            if d_prev <= d_next:
                out[s:e] = prev_val
            else:
                out[s:e] = next_val

    return out.astype(np.float32)


def load_height_profile_csv(height_csv_path: Path) -> pd.DataFrame:
    if not height_csv_path.exists():
        raise FileNotFoundError(f"Missing height profile CSV: {height_csv_path}")

    df = pd.read_csv(height_csv_path)
    required = {"x_px", "height_px"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Height profile CSV missing columns: {missing}")

    df = df.sort_values("x_px").reset_index(drop=True)
    return df


def build_event_boost_mask(
    events_df: pd.DataFrame,
    x_reference: np.ndarray,
    n_target: int,
    width_frames: int = 8,
    boost: float = 0.15,
) -> np.ndarray:
    mask = np.zeros(n_target, dtype=np.float32)
    if events_df.empty:
        return mask

    x_min = float(np.min(x_reference))
    x_max = float(np.max(x_reference))
    x_span = max(x_max - x_min, EPS)

    for _, row in events_df.iterrows():
        x_ev = float(row["x_px"])
        idx = int(round(((x_ev - x_min) / x_span) * (n_target - 1)))
        left = max(0, idx - width_frames)
        right = min(n_target, idx + width_frames + 1)

        for j in range(left, right):
            dist = abs(j - idx) / max(width_frames, 1)
            weight = max(0.0, 1.0 - dist)
            mask[j] = max(mask[j], boost * weight)

    return np.clip(mask, 0.0, 1.0).astype(np.float32)


def robust_curve_cleanup(v: np.ndarray) -> np.ndarray:
    x = np.asarray(v, dtype=np.float32)
    if len(x) == 0:
        return x

    local_med = median_filter_1d(x, SOURCE_MEDIAN_WIN).astype(np.float32)
    resid = np.abs(x - local_med)
    local_mad = median_filter_1d(resid, SOURCE_MEDIAN_WIN).astype(np.float32)
    local_scale = 1.4826 * local_mad + EPS

    z = (x - local_med) / local_scale
    spike_mask = z > SOURCE_SPIKE_THRESHOLD

    cleaned = x.copy()
    cleaned[spike_mask] = (
        SOURCE_SPIKE_BLEND * x[spike_mask]
        + (1.0 - SOURCE_SPIKE_BLEND) * local_med[spike_mask]
    ).astype(np.float32)

    return cleaned.astype(np.float32)


def find_true_runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    runs: List[Tuple[int, int]] = []
    n = len(mask)
    i = 0
    while i < n:
        if not mask[i]:
            i += 1
            continue
        start = i
        while i + 1 < n and mask[i + 1]:
            i += 1
        end = i
        runs.append((start, end))
        i += 1
    return runs


def suppress_narrow_upward_spikes(displacement: np.ndarray) -> np.ndarray:
    x = np.asarray(displacement, dtype=np.float32)
    if len(x) == 0:
        return x

    local_med = median_filter_1d(x, UPWARD_SPIKE_MEDIAN_WIN).astype(np.float32)
    delta_up = x - local_med
    candidate_mask = delta_up > UPWARD_SPIKE_THRESHOLD_PX

    repaired = x.copy()
    runs = find_true_runs(candidate_mask)

    for start, end in runs:
        width = end - start + 1
        if width <= UPWARD_SPIKE_MAX_WIDTH:
            repaired[start:end + 1] = (
                UPWARD_SPIKE_REPAIR_BLEND * repaired[start:end + 1]
                + (1.0 - UPWARD_SPIKE_REPAIR_BLEND) * local_med[start:end + 1]
            ).astype(np.float32)

    return repaired.astype(np.float32)


def _aggregate_unique_x(x_px: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x_px = np.asarray(x_px, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    order = np.argsort(x_px)
    x_px = x_px[order]
    y = y[order]

    ux = []
    uy = []

    i = 0
    n = len(x_px)
    while i < n:
        j = i + 1
        while j < n and abs(float(x_px[j]) - float(x_px[i])) < 1e-6:
            j += 1
        ux.append(float(x_px[i]))
        uy.append(float(np.median(y[i:j])))
        i = j

    return np.asarray(ux, dtype=np.float32), np.asarray(uy, dtype=np.float32)


def quantize_single_midi_to_scale(midi_val: float, root_pc: int, mode: str) -> float:
    scales = {
        "chromatic": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        "major": [0, 2, 4, 5, 7, 9, 11],
        "natural_minor": [0, 2, 3, 5, 7, 8, 10],
        "minor_pentatonic": [0, 3, 5, 7, 10],
        "major_pentatonic": [0, 2, 4, 7, 9],
        "blues_minor": [0, 3, 5, 6, 7, 10],
    }
    pcs = scales.get(mode, scales["minor_pentatonic"])

    candidates = []
    base_oct = int(np.floor(midi_val / 12.0))

    for octv in range(base_oct - 2, base_oct + 3):
        for interval in pcs:
            candidates.append(12 * octv + root_pc + interval)

    candidates = np.asarray(candidates, dtype=np.float32)
    idx = int(np.argmin(np.abs(candidates - midi_val)))
    return float(candidates[idx])


def quantize_midi_array_to_scale(midi: np.ndarray, root_pc: int, mode: str) -> np.ndarray:
    midi = np.asarray(midi, dtype=np.float32)
    out = np.zeros_like(midi, dtype=np.float32)
    for i in range(len(midi)):
        out[i] = quantize_single_midi_to_scale(float(midi[i]), root_pc=root_pc, mode=mode)
    return out.astype(np.float32)


def build_curve_pitch_driver(
    smooth_df: pd.DataFrame,
    n_target: int,
) -> np.ndarray:
    x_px = smooth_df["x_px"].to_numpy(dtype=np.float32)
    displacement = smooth_df["displacement_px"].to_numpy(dtype=np.float32)

    if len(x_px) == 0:
        return np.zeros(n_target, dtype=np.float32)

    displacement_clean = displacement.copy()
    if SUPPRESS_NARROW_UPWARD_SPIKES:
        displacement_clean = suppress_narrow_upward_spikes(displacement_clean)

    displacement_clean = robust_curve_cleanup(displacement_clean)

    pd.DataFrame({
        "x_px": x_px.astype(np.float32),
        "displacement_px_raw": displacement.astype(np.float32),
        "displacement_px_clean": displacement_clean.astype(np.float32),
    }).to_csv(OUT_CLEANED_CURVE_CSV, index=False)

    disp_base = np.percentile(displacement_clean, CURVE_BASELINE_PERCENTILE)
    disp_rel = np.clip(displacement_clean - disp_base, 0.0, None)

    max_rel = float(np.max(disp_rel))
    if max_rel <= EPS:
        disp_norm = np.zeros_like(disp_rel, dtype=np.float32)
    else:
        disp_norm = disp_rel / (max_rel + EPS)

    disp_norm = np.power(disp_norm, CURVE_BASELINE_POWER).astype(np.float32)

    if USE_PEAK_CAP and np.max(disp_norm) > EPS:
        cap_val = np.percentile(disp_norm, PEAK_CAP_PERCENTILE)
        disp_norm = np.minimum(disp_norm, cap_val)
        disp_norm = disp_norm / (np.max(disp_norm) + EPS)

    x_px_u, disp_u = _aggregate_unique_x(x_px, disp_norm)

    if len(x_px_u) == 1:
        target = np.full(n_target, disp_u[0], dtype=np.float32)
        return np.clip(target, 0.0, 1.0).astype(np.float32)

    x_norm = (x_px_u - x_px_u.min()) / max(x_px_u.max() - x_px_u.min(), EPS)

    dense_n = max(n_target * 12, len(x_norm) * 8)
    dense_t = np.linspace(0.0, 1.0, dense_n, dtype=np.float32)
    dense_curve = np.interp(dense_t, x_norm, disp_u).astype(np.float32)

    dense_curve = moving_average(dense_curve, 3).astype(np.float32)

    peak_keep = max_filter_1d(dense_curve, window=PEAK_PRESERVE_WINDOW)
    dense_curve = (
        (1.0 - PEAK_PRESERVE_BLEND) * dense_curve
        + PEAK_PRESERVE_BLEND * peak_keep
    ).astype(np.float32)

    target_t = np.linspace(0.0, 1.0, n_target, dtype=np.float32)
    target = np.interp(target_t, dense_t, dense_curve).astype(np.float32)

    return np.clip(target, 0.0, 1.0).astype(np.float32)


def build_melodic_loudness_driver(
    height_smooth_df: pd.DataFrame,
    donor_loudness: np.ndarray,
    n_target: int,
) -> np.ndarray:
    height_px = height_smooth_df["height_px"].to_numpy(dtype=np.float32)
    height_norm = normalize_01(height_px)
    height_norm = resample_1d(height_norm, n_target)

    donor_loudness = np.asarray(donor_loudness, dtype=np.float32)
    donor_norm = donor_loudness.copy()
    if np.max(donor_norm) > 0:
        donor_norm = donor_norm / (np.max(donor_norm) + EPS)

    if USE_MELODIC_LOUDNESS:
        loud_driver = (
            LOUDNESS_HEIGHT_WEIGHT * height_norm
            + LOUDNESS_DONOR_WEIGHT * donor_norm
        ).astype(np.float32)
    else:
        loud_driver = height_norm.astype(np.float32)

    loud_driver = moving_average(loud_driver, LOUDNESS_DRIVER_SMOOTH_WIN)
    loud_driver = np.clip(loud_driver, 0.0, 1.0).astype(np.float32)

    return np.clip(loud_driver, 0.0, 1.0).astype(np.float32)

# =========================================================
# BANDED PITCH -> LOUDNESS HELPERS
# =========================================================
def pitch_to_loudness_feature(pitch_hz: np.ndarray) -> np.ndarray:
    pitch_hz = np.asarray(pitch_hz, dtype=np.float32)
    feat = np.zeros_like(pitch_hz, dtype=np.float32)

    valid = np.isfinite(pitch_hz) & (pitch_hz > 0)
    if np.any(valid):
        if BANDED_LOUDNESS_USE_MIDI:
            feat[valid] = hz_to_midi(pitch_hz[valid]).astype(np.float32)
        else:
            feat[valid] = pitch_hz[valid].astype(np.float32)

    return feat.astype(np.float32)


def assign_pitch_band_from_feature(x: np.ndarray, splits: List[float]) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    bands = np.full(len(x), -1, dtype=np.int32)

    if len(splits) != 2:
        raise ValueError("BANDED_LOUDNESS_MIDI_SPLITS must contain exactly 2 values.")

    s0 = float(splits[0])
    s1 = float(splits[1])

    valid = np.isfinite(x) & (x > 0)
    bands[valid & (x < s0)] = 0
    bands[valid & (x >= s0) & (x < s1)] = 1
    bands[valid & (x >= s1)] = 2
    return bands.astype(np.int32)


def fit_simple_line(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    if len(x) < 2:
        y_mean = float(np.mean(y)) if len(y) > 0 else 0.0
        return np.asarray([y_mean, 0.0], dtype=np.float32)

    X = np.column_stack([
        np.ones_like(x, dtype=np.float32),
        x.astype(np.float32),
    ]).astype(np.float32)

    beta, _, _, _ = np.linalg.lstsq(X, y.astype(np.float32), rcond=None)
    return beta.astype(np.float32)


def fit_banded_pitch_loudness_model(
    donor_pitch_hz: np.ndarray,
    donor_loudness: np.ndarray,
) -> Dict[str, object]:
    x = pitch_to_loudness_feature(donor_pitch_hz)
    y = np.asarray(donor_loudness, dtype=np.float32)

    valid = np.isfinite(x) & np.isfinite(y) & (donor_pitch_hz > 0)
    if np.sum(valid) < 2:
        return {
            "use_midi": bool(BANDED_LOUDNESS_USE_MIDI),
            "splits": list(BANDED_LOUDNESS_MIDI_SPLITS),
            "global_beta": np.asarray([0.0, 0.0], dtype=np.float32),
            "band_betas": {
                "low": np.asarray([0.0, 0.0], dtype=np.float32),
                "mid": np.asarray([0.0, 0.0], dtype=np.float32),
                "high": np.asarray([0.0, 0.0], dtype=np.float32),
            },
            "band_counts": {
                "low": 0,
                "mid": 0,
                "high": 0,
            },
        }

    xv = x[valid]
    yv = y[valid]

    global_beta = fit_simple_line(xv, yv)
    band_ids = assign_pitch_band_from_feature(xv, BANDED_LOUDNESS_MIDI_SPLITS)

    band_betas: Dict[str, np.ndarray] = {}
    band_counts: Dict[str, int] = {}

    band_name_map = {
        0: "low",
        1: "mid",
        2: "high",
    }

    for band_id, band_name in band_name_map.items():
        mask = band_ids == band_id
        count = int(np.sum(mask))
        band_counts[band_name] = count

        if count >= BANDED_LOUDNESS_MIN_SAMPLES_PER_BAND:
            beta = fit_simple_line(xv[mask], yv[mask])
        else:
            beta = global_beta.copy()

        band_betas[band_name] = beta.astype(np.float32)

    return {
        "use_midi": bool(BANDED_LOUDNESS_USE_MIDI),
        "splits": list(BANDED_LOUDNESS_MIDI_SPLITS),
        "global_beta": global_beta.astype(np.float32),
        "band_betas": band_betas,
        "band_counts": band_counts,
    }


def predict_banded_loudness_from_pitch(
    pitch_hz: np.ndarray,
    model: Dict[str, object],
) -> np.ndarray:
    pitch_hz = np.asarray(pitch_hz, dtype=np.float32)
    x = pitch_to_loudness_feature(pitch_hz)
    band_ids = assign_pitch_band_from_feature(x, model["splits"])

    out = np.zeros_like(x, dtype=np.float32)

    band_name_map = {
        0: "low",
        1: "mid",
        2: "high",
    }

    for band_id, band_name in band_name_map.items():
        mask = band_ids == band_id
        if not np.any(mask):
            continue

        beta = np.asarray(model["band_betas"][band_name], dtype=np.float32)
        out[mask] = beta[0] + beta[1] * x[mask]

    out = np.clip(out, 0.0, 1.0).astype(np.float32)
    out[~(np.isfinite(pitch_hz) & (pitch_hz > 0))] = 0.0
    return out.astype(np.float32)

# =========================================================
# TRUE NOTE-BLOCK HELPERS
# =========================================================
def build_phrase_mask(loudness: np.ndarray, gate_on: float, gate_off: float) -> np.ndarray:
    loudness = np.asarray(loudness, dtype=np.float32)
    state = False
    mask = np.zeros_like(loudness, dtype=np.float32)

    for i, v in enumerate(loudness):
        if not state and v >= gate_on:
            state = True
        elif state and v < gate_off:
            state = False
        mask[i] = 1.0 if state else 0.0

    return mask.astype(np.float32)


def pool_midi_to_blocks(midi: np.ndarray, pool_frames: int) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    midi = np.asarray(midi, dtype=np.float32)
    n = len(midi)
    if n == 0:
        return np.zeros(0, dtype=np.float32), []

    pool_frames = max(1, int(pool_frames))
    pooled = []
    spans = []

    start = 0
    while start < n:
        end = min(n, start + pool_frames)
        seg = midi[start:end]
        pooled.append(float(np.median(seg)))
        spans.append((start, end - 1))
        start = end

    return np.asarray(pooled, dtype=np.float32), spans


def merge_short_runs(
    runs: List[Tuple[int, int]],
    vals: np.ndarray,
    min_run: int,
) -> List[Tuple[int, int]]:
    if not runs:
        return runs

    runs = list(runs)
    merged: List[Tuple[int, int]] = []
    i = 0

    while i < len(runs):
        s, e = runs[i]
        run_len = e - s + 1

        if run_len >= min_run:
            merged.append((s, e))
            i += 1
            continue

        prev_exists = len(merged) > 0
        next_exists = i + 1 < len(runs)

        if prev_exists and next_exists:
            ps, pe = merged[-1]
            ns, ne = runs[i + 1]

            prev_med = float(np.median(vals[ps:pe + 1]))
            next_med = float(np.median(vals[ns:ne + 1]))
            curr_med = float(np.median(vals[s:e + 1]))

            if abs(curr_med - prev_med) <= abs(curr_med - next_med):
                merged[-1] = (ps, e)
            else:
                runs[i + 1] = (s, ne)

        elif prev_exists:
            ps, pe = merged[-1]
            merged[-1] = (ps, e)

        elif next_exists:
            ns, ne = runs[i + 1]
            runs[i + 1] = (s, ne)

        else:
            merged.append((s, e))

        i += 1

    return merged


def segment_block_runs(
    vals: np.ndarray,
    change_thresh: float,
    min_run: int,
) -> List[Tuple[int, int]]:
    vals = np.asarray(vals, dtype=np.float32)
    n = len(vals)
    if n == 0:
        return []

    boundaries = [0]
    for i in range(1, n):
        if abs(float(vals[i] - vals[i - 1])) >= change_thresh:
            boundaries.append(i)
    boundaries.append(n)

    runs = []
    for i in range(len(boundaries) - 1):
        s = boundaries[i]
        e = boundaries[i + 1] - 1
        if e >= s:
            runs.append((s, e))

    return merge_short_runs(runs, vals, min_run=min_run)


def merge_similar_block_runs(
    runs: List[Tuple[int, int]],
    vals: np.ndarray,
    merge_thresh: float,
) -> List[Tuple[int, int]]:
    if not runs:
        return runs

    out = [runs[0]]
    for i in range(1, len(runs)):
        ps, pe = out[-1]
        cs, ce = runs[i]

        prev_med = float(np.median(vals[ps:pe + 1]))
        curr_med = float(np.median(vals[cs:ce + 1]))

        if abs(curr_med - prev_med) <= merge_thresh:
            out[-1] = (ps, ce)
        else:
            out.append((cs, ce))

    return out


def expand_pooled_runs_to_frame_runs(
    pooled_runs: List[Tuple[int, int]],
    spans: List[Tuple[int, int]],
) -> List[Tuple[int, int]]:
    frame_runs = []
    for ps, pe in pooled_runs:
        frame_runs.append((spans[ps][0], spans[pe][1]))
    return frame_runs


def rectangularize_midi_to_note_blocks(
    midi: np.ndarray,
    donor_midi: np.ndarray = None,
) -> np.ndarray:
    midi = np.asarray(midi, dtype=np.float32)
    if len(midi) == 0:
        return midi.copy()

    pooled_vals, spans = pool_midi_to_blocks(midi, NOTE_BLOCK_POOL_FRAMES)
    pooled_runs = segment_block_runs(
        pooled_vals,
        change_thresh=NOTE_BLOCK_CHANGE_THRESHOLD_SEMITONES,
        min_run=NOTE_BLOCK_MIN_RUN,
    )
    pooled_runs = merge_similar_block_runs(
        pooled_runs,
        pooled_vals,
        merge_thresh=NOTE_BLOCK_MERGE_THRESHOLD_SEMITONES,
    )
    frame_runs = expand_pooled_runs_to_frame_runs(pooled_runs, spans)

    out = np.zeros_like(midi, dtype=np.float32)

    donor_midi_arr = None
    if donor_midi is not None:
        donor_midi_arr = np.asarray(donor_midi, dtype=np.float32)

    for fs, fe in frame_runs:
        block = midi[fs:fe + 1]
        center = float(np.median(block))

        if donor_midi_arr is not None and USE_DONOR_NOTE_GUIDE and DONOR_NOTE_GUIDE_BLEND > 0:
            donor_center = float(np.median(donor_midi_arr[fs:fe + 1]))
            center = (1.0 - DONOR_NOTE_GUIDE_BLEND) * center + DONOR_NOTE_GUIDE_BLEND * donor_center

        quant_center = center
        if USE_SCALE_QUANTIZATION:
            quant_center = quantize_single_midi_to_scale(
                center,
                root_pc=QUANTIZER_ROOT_PC,
                mode=QUANTIZER_MODE,
            )
            center = (
                (1.0 - QUANTIZE_BLOCK_CENTER_BLEND) * center
                + QUANTIZE_BLOCK_CENTER_BLEND * quant_center
            )

        block_vals = np.full((fe - fs + 1,), center, dtype=np.float32)

        if BLOCK_ALLOW_WITHIN_NOTE_DEVIATION > 0 and BLOCK_MICRO_WIGGLE_BLEND > 0:
            resid = block - center
            resid = np.clip(
                resid,
                -BLOCK_ALLOW_WITHIN_NOTE_DEVIATION,
                BLOCK_ALLOW_WITHIN_NOTE_DEVIATION,
            ).astype(np.float32)
            block_vals = block_vals + BLOCK_MICRO_WIGGLE_BLEND * resid

        out[fs:fe + 1] = block_vals.astype(np.float32)

    return out.astype(np.float32)


def apply_true_note_block_pitch(
    pitch_hz_for_model: np.ndarray,
    quiet_mask: np.ndarray,
    donor_pitch_hz: np.ndarray = None,
) -> np.ndarray:
    out = np.asarray(pitch_hz_for_model, dtype=np.float32).copy()
    quiet_mask = np.asarray(quiet_mask, dtype=bool)

    voiced_mask = (~quiet_mask) & np.isfinite(out) & (out > 0)
    if np.sum(voiced_mask) == 0:
        return out.astype(np.float32)

    midi = hz_to_midi(out[voiced_mask]).astype(np.float32)

    donor_midi = None
    if donor_pitch_hz is not None:
        donor_vals = np.asarray(donor_pitch_hz, dtype=np.float32)[voiced_mask]
        valid = np.isfinite(donor_vals) & (donor_vals > 0)
        if np.any(valid):
            donor_fixed = donor_vals.copy()
            donor_fixed[~valid] = np.median(donor_fixed[valid])
            donor_midi = hz_to_midi(donor_fixed).astype(np.float32)

    midi_rect = rectangularize_midi_to_note_blocks(
        midi=midi,
        donor_midi=donor_midi,
    )

    out[voiced_mask] = midi_to_hz(midi_rect).astype(np.float32)
    out[quiet_mask] = 0.0
    return out.astype(np.float32)

# =========================================================
# CONTROL BUILDERS
# =========================================================
def build_curve_ratio_controls(
    smooth_df: pd.DataFrame,
    events_df: pd.DataFrame,
    height_smooth_df: pd.DataFrame,
    donor_pitch_hz: np.ndarray,
    donor_loudness: np.ndarray,
    banded_loudness_model: Dict[str, object] = None,
) -> pd.DataFrame:
    n_target = len(donor_pitch_hz)

    pitch_driver = build_curve_pitch_driver(
        smooth_df=smooth_df,
        n_target=n_target,
    )

    donor_pitch_hz = np.asarray(donor_pitch_hz, dtype=np.float32)
    donor_loudness = np.asarray(donor_loudness, dtype=np.float32)

    loud_driver = build_melodic_loudness_driver(
        height_smooth_df=height_smooth_df,
        donor_loudness=donor_loudness,
        n_target=n_target,
    )

    curve_pitch_abs = CURVE_PITCH_MIN_HZ + pitch_driver * (CURVE_PITCH_MAX_HZ - CURVE_PITCH_MIN_HZ)
    curve_pitch_abs = curve_pitch_abs.astype(np.float32)

    if APPLY_CURVE_PITCH_SMOOTHING and CURVE_PITCH_SMOOTH_WIN > 1:
        curve_pitch_abs = moving_average(curve_pitch_abs, CURVE_PITCH_SMOOTH_WIN).astype(np.float32)

    pitch_hz_raw = (
        DONOR_PITCH_WEIGHT * donor_pitch_hz
        + CURVE_PITCH_WEIGHT * curve_pitch_abs
    ).astype(np.float32)

    loud_ratio = HEIGHT_LOUDNESS_RATIO_MIN + loud_driver * (HEIGHT_LOUDNESS_RATIO_MAX - HEIGHT_LOUDNESS_RATIO_MIN)
    loud_ratio = np.clip(
        loud_ratio,
        HEIGHT_LOUDNESS_RATIO_MIN,
        HEIGHT_LOUDNESS_RATIO_MAX
    ).astype(np.float32)

    event_boost = np.zeros(n_target, dtype=np.float32)
    if USE_EVENT_ATTACK_BOOST:
        event_boost = build_event_boost_mask(
            events_df=events_df,
            x_reference=smooth_df["x_px"].to_numpy(dtype=np.float32),
            n_target=n_target,
            width_frames=EVENT_ATTACK_WIDTH_FRAMES,
            boost=EVENT_ATTACK_BOOST,
        )
        loud_ratio = loud_ratio * (1.0 + event_boost)
        loud_ratio = np.clip(
            loud_ratio,
            HEIGHT_LOUDNESS_RATIO_MIN,
            HEIGHT_LOUDNESS_RATIO_MAX + EVENT_ATTACK_BOOST,
        )

    # -----------------------------------------------------
    # BASELINE LOUDNESS
    # -----------------------------------------------------
    loudness_base = donor_loudness * (1.0 + LOUDNESS_BLEND * (loud_ratio - 1.0))
    loudness_base = np.clip(loudness_base, 0.0, 1.0).astype(np.float32)

    baseline_quiet = loudness_base <= LOUDNESS_GATE_THRESHOLD

    invalid_pitch = (
        ~np.isfinite(pitch_hz_raw)
        | (pitch_hz_raw < CURVE_PITCH_MIN_HZ * 0.75)
        | (pitch_hz_raw > CURVE_PITCH_MAX_HZ * 1.35)
    )
    pitch_hz_raw[invalid_pitch] = 0.0
    pitch_hz_raw[baseline_quiet] = 0.0

    # -----------------------------------------------------
    # NEW: nearest fill instead of linear fill
    # -----------------------------------------------------
    if np.any(pitch_hz_raw > 0):
        if USE_NEAREST_PITCH_FILL:
            pitch_hz_for_model = fill_unvoiced_pitch_nearest(pitch_hz_raw)
        else:
            pitch_hz_for_model = fill_unvoiced_pitch_linear(pitch_hz_raw)
    else:
        pitch_hz_for_model = np.full((n_target,), 220.0, dtype=np.float32)

    # -----------------------------------------------------
    # NEW: collapse short in-between transition runs
    # -----------------------------------------------------
    if USE_TRANSITION_COLLAPSE:
        pitch_hz_for_model = collapse_short_pitch_transitions(
            pitch_hz=pitch_hz_for_model,
            min_run_frames=TRANSITION_MIN_RUN_FRAMES,
            jump_threshold_semitones=TRANSITION_JUMP_THRESHOLD_SEMITONES,
            zero_ambiguous=TRANSITION_ZERO_AMBIGUOUS,
            zero_max_run_frames=TRANSITION_ZERO_MAX_RUN_FRAMES,
        )

    if USE_TRUE_NOTE_BLOCKS:
        pitch_hz_for_model = apply_true_note_block_pitch(
            pitch_hz_for_model=pitch_hz_for_model,
            quiet_mask=baseline_quiet,
            donor_pitch_hz=donor_pitch_hz,
        )

    pitch_hz_for_model[baseline_quiet] = 0.0

    # -----------------------------------------------------
    # BANDED PITCH->LOUDNESS MODEL
    # -----------------------------------------------------
    if USE_BANDED_PITCH_LOUDNESS and banded_loudness_model is not None:
        loudness_regression = predict_banded_loudness_from_pitch(
            pitch_hz=pitch_hz_for_model,
            model=banded_loudness_model,
        )
        loudness_for_model = (
            (1.0 - BANDED_LOUDNESS_BLEND) * loudness_base
            + BANDED_LOUDNESS_BLEND * loudness_regression
        ).astype(np.float32)
    else:
        loudness_regression = np.zeros(n_target, dtype=np.float32)
        loudness_for_model = loudness_base.astype(np.float32)

    loudness_for_model = np.clip(loudness_for_model, 0.0, 1.0).astype(np.float32)

    if USE_ATTACK_RELEASE_LOUDNESS:
        loudness_for_model = attack_release_filter(
            loudness_for_model,
            attack=LOUDNESS_ATTACK_COEF,
            release=LOUDNESS_RELEASE_COEF,
        )
        loudness_for_model = np.clip(loudness_for_model, 0.0, 1.0).astype(np.float32)

    if USE_PHRASE_GATE:
        phrase_mask = build_phrase_mask(
            loudness_for_model,
            gate_on=PHRASE_GATE_ON,
            gate_off=PHRASE_GATE_OFF,
        )
        loudness_for_model = (loudness_for_model * phrase_mask).astype(np.float32)
    else:
        phrase_mask = np.ones(n_target, dtype=np.float32)

    quiet = loudness_for_model <= LOUDNESS_GATE_THRESHOLD
    pitch_hz_for_model[quiet] = 0.0
    loudness_for_model[quiet] = 0.0
    loudness_regression[quiet] = 0.0
    loudness_base[quiet] = 0.0

    df = pd.DataFrame({
        "frame_idx": np.arange(n_target, dtype=np.int32),
        "time_sec": np.arange(n_target, dtype=np.float32) / float(FRAME_RATE),
        "donor_pitch_hz": donor_pitch_hz.astype(np.float32),
        "donor_loudness": donor_loudness.astype(np.float32),
        "curve_pitch_driver_norm": pitch_driver.astype(np.float32),
        "height_loud_driver_norm": loud_driver.astype(np.float32),
        "curve_pitch_abs_hz": curve_pitch_abs.astype(np.float32),
        "event_attack_boost": event_boost.astype(np.float32),
        "phrase_mask": phrase_mask.astype(np.float32),
        "pitch_hz_raw": pitch_hz_raw.astype(np.float32),
        "pitch_hz_for_model": pitch_hz_for_model.astype(np.float32),
        "loudness_base": loudness_base.astype(np.float32),
        "loudness_regression": loudness_regression.astype(np.float32),
        "loudness_for_model": loudness_for_model.astype(np.float32),
        "loudness_ratio": loud_ratio.astype(np.float32),
        "sr": np.full(n_target, SR, dtype=np.int32),
        "hop_length": np.full(n_target, HOP_LENGTH, dtype=np.int32),
    })
    return df


def build_pitch_loudness_bundle(curve_ratio_df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "frame_idx": curve_ratio_df["frame_idx"].to_numpy(dtype=np.int32),
        "sr": curve_ratio_df["sr"].to_numpy(dtype=np.int32),
        "hop_length": curve_ratio_df["hop_length"].to_numpy(dtype=np.int32),
        "pitch_hz_for_model": curve_ratio_df["pitch_hz_for_model"].to_numpy(dtype=np.float32),
        "loudness_for_model": curve_ratio_df["loudness_for_model"].to_numpy(dtype=np.float32),
    })


def build_audio_based_controls(
    controls_df: pd.DataFrame,
    audio_reference_controls: Dict[str, np.ndarray],
    target_num_harmonics: int,
    target_num_noise_bands: int,
    donor_harmonic_blend: float = AUDIO_BASED_DONOR_HARMONIC_BLEND,
    apply_curve_brightness_tilt: bool = AUDIO_BASED_APPLY_CURVE_BRIGHTNESS_TILT,
    brightness_tilt_strength: float = AUDIO_BASED_BRIGHTNESS_TILT_STRENGTH,
    donor_noise_blend: float = AUDIO_BASED_DONOR_NOISE_BLEND,
) -> Dict[str, np.ndarray]:
    n_target = len(controls_df)

    pitch_hz_for_model = controls_df["pitch_hz_for_model"].to_numpy(dtype=np.float32)
    loudness_for_model = controls_df["loudness_for_model"].to_numpy(dtype=np.float32)

    ref_amp = np.asarray(audio_reference_controls["amplitudes"], dtype=np.float32).reshape(-1, 1)
    ref_harm = np.asarray(audio_reference_controls["harmonics"], dtype=np.float32)
    ref_noise = audio_reference_controls["noise"]
    ref_noise = None if ref_noise is None else np.asarray(ref_noise, dtype=np.float32)

    ref_amp = resample_2d_rows(ref_amp, n_target)
    ref_harm = resample_2d_rows(ref_harm, n_target)
    ref_harm = resize_control_width(ref_harm, target_num_harmonics, fill_mode="tile")

    if ref_noise is not None and target_num_noise_bands > 0:
        ref_noise = resample_2d_rows(ref_noise, n_target)
        ref_noise = resize_control_width(ref_noise, target_num_noise_bands, fill_mode="tile")
    else:
        ref_noise = None

    rb = build_formula_based_controls(
        controls_df=controls_df,
        num_harmonics=target_num_harmonics,
        num_noise_bands=target_num_noise_bands,
    )

    formula_amp = np.asarray(rb["amplitudes"], dtype=np.float32).reshape(-1, 1)
    formula_harm = np.asarray(rb["harmonics"], dtype=np.float32)
    formula_noise = rb["noise"]
    formula_noise = None if formula_noise is None else np.asarray(formula_noise, dtype=np.float32)

    amplitudes = formula_amp.astype(np.float32)

    if AUDIO_BASED_USE_DIRECT_AMPLITUDE:
        amplitudes = ref_amp.astype(np.float32)

    if AUDIO_BASED_MODULATE_AMPLITUDE_BY_INPUT:
        ref_amp_norm = ref_amp[:, 0].copy()
        if np.max(ref_amp_norm) > 0:
            ref_amp_norm = ref_amp_norm / (np.max(ref_amp_norm) + EPS)
        amplitudes[:, 0] *= (0.5 + 0.5 * ref_amp_norm)

    amplitudes[:, 0] = moving_average(amplitudes[:, 0], AMPLITUDE_SMOOTH_WIN)
    amplitudes[:, 0] = np.clip(amplitudes[:, 0], 0.0, 1.0)

    if AUDIO_BASED_USE_DIRECT_HARMONICS:
        donor_w = float(np.clip(donor_harmonic_blend, 0.0, 1.0))
        formula_w = 1.0 - donor_w
        harmonics = donor_w * ref_harm.astype(np.float32) + formula_w * formula_harm.astype(np.float32)
    else:
        harmonics = formula_harm.astype(np.float32)

    harmonics = harmonics / (np.sum(harmonics, axis=1, keepdims=True) + EPS)

    if apply_curve_brightness_tilt:
        if "curve_pitch_driver_norm" in controls_df.columns and "height_loud_driver_norm" in controls_df.columns:
            pitch_driver = controls_df["curve_pitch_driver_norm"].to_numpy(dtype=np.float32)
            loud_driver = controls_df["height_loud_driver_norm"].to_numpy(dtype=np.float32)
            brightness = 0.5 * pitch_driver + 0.5 * loud_driver
        else:
            brightness = normalize_01(loudness_for_model)

        brightness = np.clip(brightness, 0.0, 1.0).astype(np.float32)

        harm_idx = np.arange(target_num_harmonics, dtype=np.float32)
        denom = max(float(target_num_harmonics - 1), 1.0)
        harm_pos = harm_idx / denom

        for t in range(harmonics.shape[0]):
            b = float(brightness[t])
            tilt = 1.0 + brightness_tilt_strength * b * harm_pos
            tilt[0] *= 1.05
            if target_num_harmonics > 1:
                tilt[1] *= 1.02
            harmonics[t] *= tilt.astype(np.float32)
            harmonics[t] /= (np.sum(harmonics[t]) + EPS)

    if AUDIO_BASED_USE_DIRECT_NOISE and ref_noise is not None:
        if formula_noise is not None:
            donor_noise_blend = float(np.clip(donor_noise_blend, 0.0, 1.0))
            formula_noise_blend = 1.0 - donor_noise_blend
            noise = donor_noise_blend * ref_noise.astype(np.float32) + formula_noise_blend * formula_noise.astype(np.float32)
        else:
            noise = ref_noise.astype(np.float32)
    else:
        noise = formula_noise.astype(np.float32) if formula_noise is not None else None

    if noise is not None:
        noise = np.clip(noise, 0.0, 1.0)

    quiet = loudness_for_model <= LOUDNESS_GATE_THRESHOLD
    amplitudes[quiet, 0] = 0.0
    harmonics[quiet] = 0.0
    harmonics[quiet, 0] = 1.0

    if noise is not None:
        noise[quiet] = 0.0

    return {
        "pitch_hz_for_model": pitch_hz_for_model.astype(np.float32),
        "loudness_for_model": loudness_for_model.astype(np.float32),
        "amplitudes": amplitudes.astype(np.float32),
        "harmonics": harmonics.astype(np.float32),
        "noise": noise,
    }

# =========================================================
# VIS HELPERS
# =========================================================
def compact_to_long_timeseries(
    compact_df: pd.DataFrame,
    mode_name: str,
    top_k_harmonics: int = 32,
    top_k_noise: int = 16,
) -> pd.DataFrame:
    compact_df = enforce_dense_compact_layout(compact_df)

    harmonic_cols = sort_feature_cols(list(compact_df.columns), "harmonic_")
    noise_cols = sort_feature_cols(list(compact_df.columns), "noise_")

    parts = []

    amp_df = compact_df[["frame_idx", "amplitudes"]].copy()
    amp_df = amp_df.rename(columns={"amplitudes": "value"})
    amp_df["mode"] = mode_name
    amp_df["component_type"] = "amplitude"
    amp_df["component_name"] = "amplitudes"
    parts.append(amp_df)

    pitch_df = compact_df[["frame_idx", "pitch_hz_for_model"]].copy()
    pitch_df = pitch_df.rename(columns={"pitch_hz_for_model": "value"})
    pitch_df["mode"] = mode_name
    pitch_df["component_type"] = "pitch"
    pitch_df["component_name"] = "pitch_hz_for_model"
    parts.append(pitch_df)

    loud_df = compact_df[["frame_idx", "loudness_for_model"]].copy()
    loud_df = loud_df.rename(columns={"loudness_for_model": "value"})
    loud_df["mode"] = mode_name
    loud_df["component_type"] = "loudness"
    loud_df["component_name"] = "loudness_for_model"
    parts.append(loud_df)

    if harmonic_cols:
        harm_arr = compact_df[harmonic_cols].to_numpy(dtype=np.float32)
        harm_mean = harm_arr.mean(axis=0)
        top_h_idx = np.argsort(harm_mean)[::-1][:min(top_k_harmonics, len(harmonic_cols))]
        selected_h_cols = [harmonic_cols[i] for i in top_h_idx]

        harm_melt = compact_df[["frame_idx"] + selected_h_cols].melt(
            id_vars="frame_idx",
            value_vars=selected_h_cols,
            var_name="component_name",
            value_name="value",
        )
        harm_melt["mode"] = mode_name
        harm_melt["component_type"] = "harmonic"
        parts.append(harm_melt)

    if noise_cols:
        noise_arr = compact_df[noise_cols].to_numpy(dtype=np.float32)
        noise_mean = noise_arr.mean(axis=0)
        top_n_idx = np.argsort(noise_mean)[::-1][:min(top_k_noise, len(noise_cols))]
        selected_n_cols = [noise_cols[i] for i in top_n_idx]

        noise_melt = compact_df[["frame_idx"] + selected_n_cols].melt(
            id_vars="frame_idx",
            value_vars=selected_n_cols,
            var_name="component_name",
            value_name="value",
        )
        noise_melt["mode"] = mode_name
        noise_melt["component_type"] = "noise"
        parts.append(noise_melt)

    long_df = pd.concat(parts, axis=0, ignore_index=True)
    long_df["time_sec"] = long_df["frame_idx"].astype(np.float32) / float(FRAME_RATE)

    return long_df[["mode", "frame_idx", "time_sec", "component_type", "component_name", "value"]]


def build_all_modes_long_df(mode_to_compact_df: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    parts = []
    for mode_name, compact_df in mode_to_compact_df.items():
        parts.append(compact_to_long_timeseries(compact_df=compact_df, mode_name=mode_name))
    return pd.concat(parts, axis=0, ignore_index=True)


def plot_summary(curve_ratio_df: pd.DataFrame, out_path: Path) -> None:
    t = curve_ratio_df["time_sec"].to_numpy(dtype=np.float32)

    fig, ax = plt.subplots(6, 1, figsize=(14, 16), sharex=True)

    ax[0].plot(t, curve_ratio_df["donor_loudness"], label="donor_loudness", linewidth=1.0)
    ax[0].plot(t, curve_ratio_df["loudness_base"], label="loudness_base", linewidth=1.0, alpha=0.85)
    ax[0].plot(t, curve_ratio_df["loudness_regression"], label="loudness_regression", linewidth=1.0, alpha=0.85)
    ax[0].plot(t, curve_ratio_df["loudness_for_model"], label="loudness_for_model", linewidth=1.2)
    ax[0].set_title("Loudness: donor vs base vs regression vs final")
    ax[0].legend()
    ax[0].grid(alpha=0.3)

    ax[1].plot(t, curve_ratio_df["curve_pitch_abs_hz"], label="curve_pitch_abs_hz", linewidth=1.0)
    ax[1].plot(t, curve_ratio_df["pitch_hz_raw"], label="pitch_hz_raw", linewidth=1.0, alpha=0.7)
    ax[1].plot(t, curve_ratio_df["pitch_hz_for_model"], label="pitch_hz_for_model", linewidth=1.6, alpha=0.9)
    ax[1].set_title("Pitch: curve-driven absolute pitch vs final pitch")
    ax[1].legend()
    ax[1].grid(alpha=0.3)

    ax[2].plot(t, curve_ratio_df["curve_pitch_driver_norm"], label="curve_pitch_driver_norm", linewidth=1.0)
    ax[2].plot(t, curve_ratio_df["height_loud_driver_norm"], label="height_loud_driver_norm", linewidth=1.0)
    ax[2].set_title("Normalized pitch/loudness drivers")
    ax[2].legend()
    ax[2].grid(alpha=0.3)

    ax[3].plot(t, curve_ratio_df["loudness_ratio"], label="loudness_ratio", linewidth=1.0)
    ax[3].set_title("Height-derived loudness ratio")
    ax[3].legend()
    ax[3].grid(alpha=0.3)

    ax[4].plot(t, curve_ratio_df["event_attack_boost"], label="event_attack_boost", linewidth=1.0)
    ax[4].set_title("Event-based transient boost")
    ax[4].legend()
    ax[4].grid(alpha=0.3)

    if "phrase_mask" in curve_ratio_df.columns:
        ax[5].plot(t, curve_ratio_df["phrase_mask"], label="phrase_mask", linewidth=1.0)
        ax[5].set_title("Phrase gate mask")
        ax[5].legend()
        ax[5].grid(alpha=0.3)

    ax[5].set_xlabel("Time (sec)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_mode_amplitudes(mode_to_compact_df: Dict[str, pd.DataFrame], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 5))

    for mode_name, df in mode_to_compact_df.items():
        t = df["frame_idx"].to_numpy(dtype=np.float32) / float(FRAME_RATE)
        amp = df["amplitudes"].to_numpy(dtype=np.float32)
        ax.plot(t, amp, linewidth=1.0, label=mode_name)

    ax.set_title("Amplitude trajectories by mode")
    ax.set_xlabel("Time (sec)")
    ax.set_ylabel("Amplitude")
    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_json(path: Path, payload: Dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2))

# =========================================================
# MAIN
# =========================================================
def main() -> None:
    ensure_dir(OUTPUT_DIR)

    print("=" * 100)
    print("IMAGE CURVE + HEIGHT PROFILE -> RECTANGULAR NOTE PITCH + DONOR TIMBRE -> 3 AUDIO OUTPUT MODES")
    print("=" * 100)

    print("\n[1/7] Extracting curve and height profile from image...")
    raw_df, smooth_df, simplified_df, events_df = run_pipeline(
        image_path=str(IMAGE_PATH),
        output_dir=str(OUTPUT_DIR),
        zero_mode=ZERO_MODE,
        zero_line_y=ZERO_LINE_Y,
        zero_p1=ZERO_P1,
        zero_p2=ZERO_P2,
        preprocess_mode=PREPROCESS_MODE,
        gaussian_kernel=GAUSSIAN_KERNEL,
        canny_low=CANNY_LOW,
        canny_high=CANNY_HIGH,
        boundary_mode=BOUNDARY_MODE,
        use_skeleton=USE_SKELETON,
        min_curve_points=MIN_CURVE_POINTS,
        min_run=MIN_RUN,
        spike_max_jump=SPIKE_MAX_JUMP,
        smooth_window=SMOOTH_WINDOW,
        smooth_method=SMOOTH_METHOD,
        smooth_polyorder=SMOOTH_POLYORDER,
        simplify_step=SIMPLIFY_STEP,
        peak_prominence=PEAK_PROMINENCE,
    )

    raw_df.to_csv(OUT_RAW_CURVE_CSV, index=False)
    smooth_df.to_csv(OUT_SMOOTH_CURVE_CSV, index=False)
    simplified_df.to_csv(OUT_SIMPLIFIED_CURVE_CSV, index=False)
    events_df.to_csv(OUT_EVENTS_CSV, index=False)

    height_smooth_df = load_height_profile_csv(OUT_HEIGHT_SMOOTH_CSV)

    print("[2/7] Analyzing donor/input audio...")
    y = load_audio_mono(INPUT_WAV, sr=SR)

    s_complex, audio_attrs, audio_meta = compute_framewise_attributes(
        y=y,
        input_wav=INPUT_WAV,
        ckpt_path=CKPT_PATH,
        sr=SR,
    )

    audio_reference_controls = estimate_compact_controls_from_audio(
        s_complex=s_complex,
        attrs=audio_attrs,
        meta=audio_meta,
        num_harmonics=NUM_HARMONICS,
        num_noise_bands=NUM_NOISE_BANDS,
    )

    audio_reference_compact_df = build_compact_dataframe_from_controls(
        compact_controls=audio_reference_controls,
        meta=audio_meta,
    )
    audio_reference_compact_df = maybe_round_df(audio_reference_compact_df)
    audio_reference_compact_df.to_csv(OUT_AUDIO_REFERENCE_COMPACT_CSV, index=False)

    save_json(
        OUT_AUDIO_REFERENCE_FULL_META_JSON,
        {
            "input_wav": str(INPUT_WAV),
            "checkpoint": str(CKPT_PATH),
            "audio_frames": int(audio_meta.n_frames),
            "duration_sec": float(audio_meta.duration_sec),
            "num_harmonics": int(NUM_HARMONICS),
            "num_noise_bands": int(NUM_NOISE_BANDS),
        },
    )

    print("[3/7] Fitting banded donor pitch->loudness model...")
    banded_loudness_model = fit_banded_pitch_loudness_model(
        donor_pitch_hz=audio_reference_controls["pitch_hz_for_model"],
        donor_loudness=audio_reference_controls["loudness_for_model"],
    )

    print("[4/7] Building rectangular note pitch + loudness controls...")
    curve_ratio_df = build_curve_ratio_controls(
        smooth_df=smooth_df,
        events_df=events_df,
        height_smooth_df=height_smooth_df,
        donor_pitch_hz=audio_reference_controls["pitch_hz_for_model"],
        donor_loudness=audio_reference_controls["loudness_for_model"],
        banded_loudness_model=banded_loudness_model,
    )
    curve_ratio_df = maybe_round_df(curve_ratio_df)
    curve_ratio_df.to_csv(OUT_RATIO_CSV, index=False)

    pitch_loudness_df = build_pitch_loudness_bundle(curve_ratio_df)
    pitch_loudness_df = maybe_round_df(pitch_loudness_df)
    pitch_loudness_df.to_csv(OUT_PITCH_LOUDNESS_CSV, index=False)

    curve_meta = AnalysisMeta(
        sr=SR,
        duration_sec=float(len(curve_ratio_df)) / float(FRAME_RATE),
        num_samples=int(len(curve_ratio_df) * HOP_LENGTH),
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        window=WINDOW,
        n_frames=int(len(curve_ratio_df)),
        n_freq_bins=(N_FFT // 2 + 1),
        frame_rate=FRAME_RATE,
        source_wav=str(IMAGE_PATH),
        ckpt_path=str(CKPT_PATH),
    )

    print("[5/7] Running synthesis branches...")
    if GENERATION_MODE == "all":
        modes = ["formula_based", "ml_based", "audio_based"]
    else:
        modes = [GENERATION_MODE]

    mode_to_compact_df: Dict[str, pd.DataFrame] = {}
    mode_to_csv_path: Dict[str, str] = {}
    mode_to_wav_path: Dict[str, str] = {}

    model = None

    for mode_name in modes:
        print(f"  -> {mode_name}")

        if mode_name == "formula_based":
            compact_controls = build_formula_based_controls(
                controls_df=curve_ratio_df,
                num_harmonics=NUM_HARMONICS,
                num_noise_bands=NUM_NOISE_BANDS,
            )

        elif mode_name == "ml_based":
            if model is None:
                model = load_audio_model(CKPT_PATH).to(DEVICE)

            compact_controls = predict_synth_controls_from_pitch_loudness(
                model=model,
                pitch_hz_for_model=curve_ratio_df["pitch_hz_for_model"].to_numpy(dtype=np.float32),
                loudness_for_model=curve_ratio_df["loudness_for_model"].to_numpy(dtype=np.float32),
            )

            compact_controls["harmonics"] = resize_control_width(
                compact_controls["harmonics"],
                NUM_HARMONICS,
                fill_mode="tile",
            )
            compact_controls["noise"] = (
                resize_control_width(compact_controls["noise"], NUM_NOISE_BANDS, fill_mode="tile")
                if (compact_controls["noise"] is not None and NUM_NOISE_BANDS > 0)
                else None
            )

        elif mode_name == "audio_based":
            compact_controls = build_audio_based_controls(
                controls_df=curve_ratio_df,
                audio_reference_controls=audio_reference_controls,
                target_num_harmonics=NUM_HARMONICS,
                target_num_noise_bands=NUM_NOISE_BANDS,
                donor_harmonic_blend=AUDIO_BASED_DONOR_HARMONIC_BLEND,
                apply_curve_brightness_tilt=AUDIO_BASED_APPLY_CURVE_BRIGHTNESS_TILT,
                brightness_tilt_strength=AUDIO_BASED_BRIGHTNESS_TILT_STRENGTH,
                donor_noise_blend=AUDIO_BASED_DONOR_NOISE_BLEND,
            )
        else:
            raise ValueError(f"Unknown mode: {mode_name}")

        compact_df = build_compact_dataframe_from_controls(
            compact_controls=compact_controls,
            meta=curve_meta,
        )
        compact_df = enforce_dense_compact_layout(compact_df)
        compact_df = maybe_round_df(compact_df)

        out_csv = OUTPUT_DIR / f"{RUN_PREFIX}_{mode_name}.csv"
        out_wav = OUTPUT_DIR / f"{RUN_PREFIX}_{mode_name}.wav"

        compact_df.to_csv(out_csv, index=False)
        reconstruct_from_compact_csv(
            df=compact_df,
            out_wav=out_wav,
            noise_scale=DEFAULT_NOISE_SCALE,
        )

        mode_to_compact_df[mode_name] = compact_df
        mode_to_csv_path[mode_name] = str(out_csv)
        mode_to_wav_path[mode_name] = str(out_wav)

    long_df = build_all_modes_long_df(mode_to_compact_df)
    long_df = maybe_round_df(long_df)
    long_df.to_csv(OUT_LONG_VIS_CSV, index=False)

    print("[6/7] Saving plots...")
    plot_summary(curve_ratio_df, OUT_SUMMARY_PLOT)
    plot_mode_amplitudes(mode_to_compact_df, OUT_COMPONENTS_PLOT)

    print("[7/7] Saving metadata...")
    save_json(
        OUT_META_JSON,
        {
            "image_path": str(IMAGE_PATH),
            "input_wav": str(INPUT_WAV),
            "checkpoint": str(CKPT_PATH),
            "generation_mode": GENERATION_MODE,
            "audio_frames": int(audio_meta.n_frames),
            "curve_points": int(len(smooth_df)),
            "height_points": int(len(height_smooth_df)),
            "curve_pitch_min_hz": float(CURVE_PITCH_MIN_HZ),
            "curve_pitch_max_hz": float(CURVE_PITCH_MAX_HZ),
            "donor_pitch_weight": float(DONOR_PITCH_WEIGHT),
            "curve_pitch_weight": float(CURVE_PITCH_WEIGHT),
            "curve_pitch_smooth_win": int(CURVE_PITCH_SMOOTH_WIN),
            "use_true_note_blocks": bool(USE_TRUE_NOTE_BLOCKS),
            "note_block_pool_frames": int(NOTE_BLOCK_POOL_FRAMES),
            "note_block_change_threshold_semitones": float(NOTE_BLOCK_CHANGE_THRESHOLD_SEMITONES),
            "note_block_min_run": int(NOTE_BLOCK_MIN_RUN),
            "note_block_merge_threshold_semitones": float(NOTE_BLOCK_MERGE_THRESHOLD_SEMITONES),
            "use_scale_quantization": bool(USE_SCALE_QUANTIZATION),
            "quantizer_mode": QUANTIZER_MODE,
            "quantizer_root_pc": int(QUANTIZER_ROOT_PC),
            "quantize_block_center_blend": float(QUANTIZE_BLOCK_CENTER_BLEND),
            "block_allow_within_note_deviation": float(BLOCK_ALLOW_WITHIN_NOTE_DEVIATION),
            "block_micro_wiggle_blend": float(BLOCK_MICRO_WIGGLE_BLEND),
            "use_nearest_pitch_fill": bool(USE_NEAREST_PITCH_FILL),
            "use_transition_collapse": bool(USE_TRANSITION_COLLAPSE),
            "transition_min_run_frames": int(TRANSITION_MIN_RUN_FRAMES),
            "transition_jump_threshold_semitones": float(TRANSITION_JUMP_THRESHOLD_SEMITONES),
            "transition_zero_ambiguous": bool(TRANSITION_ZERO_AMBIGUOUS),
            "transition_zero_max_run_frames": int(TRANSITION_ZERO_MAX_RUN_FRAMES),
            "use_banded_pitch_loudness": bool(USE_BANDED_PITCH_LOUDNESS),
            "banded_loudness_blend": float(BANDED_LOUDNESS_BLEND),
            "banded_loudness_use_midi": bool(BANDED_LOUDNESS_USE_MIDI),
            "banded_loudness_midi_splits": list(BANDED_LOUDNESS_MIDI_SPLITS),
            "banded_loudness_min_samples_per_band": int(BANDED_LOUDNESS_MIN_SAMPLES_PER_BAND),
            "banded_loudness_global_beta": [float(x) for x in banded_loudness_model["global_beta"]],
            "banded_loudness_band_betas": {
                k: [float(v[0]), float(v[1])]
                for k, v in banded_loudness_model["band_betas"].items()
            },
            "banded_loudness_band_counts": {
                k: int(v) for k, v in banded_loudness_model["band_counts"].items()
            },
            "mode_csv_paths": mode_to_csv_path,
            "mode_wav_paths": mode_to_wav_path,
            "curve_ratio_csv": str(OUT_RATIO_CSV),
            "pitch_loudness_csv": str(OUT_PITCH_LOUDNESS_CSV),
            "summary_plot": str(OUT_SUMMARY_PLOT),
            "components_plot": str(OUT_COMPONENTS_PLOT),
            "components_long_csv": str(OUT_LONG_VIS_CSV),
            "height_profile_smooth_csv": str(OUT_HEIGHT_SMOOTH_CSV),
            "cleaned_curve_csv": str(OUT_CLEANED_CURVE_CSV),
        },
    )

    print("Done.\n")
    print("Outputs:")
    print(f"  Curve ratio CSV       : {OUT_RATIO_CSV}")
    print(f"  Pitch+loudness CSV    : {OUT_PITCH_LOUDNESS_CSV}")
    print(f"  Height smooth CSV     : {OUT_HEIGHT_SMOOTH_CSV}")
    print(f"  Cleaned curve CSV     : {OUT_CLEANED_CURVE_CSV}")
    for mode_name in mode_to_csv_path:
        print(f"  {mode_name} CSV        : {mode_to_csv_path[mode_name]}")
        print(f"  {mode_name} WAV        : {mode_to_wav_path[mode_name]}")
    print(f"  Components long CSV   : {OUT_LONG_VIS_CSV}")


if __name__ == "__main__":
    main()