import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import matplotlib.pyplot as plt

# =========================================================
# PATH SETUP
# =========================================================
HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[2]          # experiments -> src -> core
PROJECT_ROOT = REPO_ROOT.parent      # Music_AI_1.1

sys.path.append(str(REPO_ROOT))

from src.models.train_instrument import AudioSynthTrainer
from src.models.signal_processing import harmonic_synthesis, noise_synthesis

# =========================================================
# USER CONFIG
# =========================================================
RUN_MODE = "generate"
# choices:
#   "generate"
#   "reconstruct_compact_csv"
#   "reconstruct_pitch_loudness_csv"

GENERATION_MODE = "all"
# choices:
#   "formula_based"
#   "ml_based"
#   "audio_based"
#   "all"

# ---------------------------------------------------------
# INPUTS
# ---------------------------------------------------------
CURVE_SIMPLE_ROOT = Path(r"C:\curve_simple")
CURVE_OUTPUT_DIR = CURVE_SIMPLE_ROOT / "output"
CURVE_CSV_PATH = CURVE_OUTPUT_DIR / "curve_mapping_smooth.csv"

# Internal donor/reference audio used only for audio_based mode
INPUT_WAV = Path(r"C:\curve_simple\input\input.wav")

OUTPUT_DIR = Path(r"C:\curve_simple\output\curve_compact_generation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CKPT_PATH = Path(r"C:\curve_simple\checkpoints\Guitar.ckpt")

RUN_PREFIX = "curve_self_sufficient"

ROUND_FLOATS_FOR_CSV = False
CSV_FLOAT_DECIMALS = 6

# =========================================================
# OUTPUT FILES
# =========================================================
OUT_CURVE_CONTROLS_CSV = OUTPUT_DIR / f"{RUN_PREFIX}_curve_controls.csv"
OUT_PITCH_LOUDNESS_CSV = OUTPUT_DIR / f"{RUN_PREFIX}_pitch_loudness_bundle.csv"

OUT_AUDIO_ANALYSIS_COMPACT_CSV = OUTPUT_DIR / f"{RUN_PREFIX}_audio_reference_compact_bundle.csv"
OUT_AUDIO_ANALYSIS_FULL_CSV = OUTPUT_DIR / f"{RUN_PREFIX}_audio_reference_full_bundle.csv"

OUT_LONG_VIS_CSV = OUTPUT_DIR / f"{RUN_PREFIX}_components_long.csv"
OUT_SUMMARY_PLOT = OUTPUT_DIR / f"{RUN_PREFIX}_summary_plot.png"
OUT_COMPONENTS_PLOT = OUTPUT_DIR / f"{RUN_PREFIX}_components_plot.png"
OUT_META_JSON = OUTPUT_DIR / f"{RUN_PREFIX}_meta.json"

RECON_COMPACT_WAV = OUTPUT_DIR / f"{RUN_PREFIX}_reconstructed_from_compact_csv.wav"
RECON_PITCH_LOUDNESS_WAV = OUTPUT_DIR / f"{RUN_PREFIX}_reconstructed_from_pitch_loudness_csv.wav"

# =========================================================
# AUDIO / ANALYSIS CONFIG
# =========================================================
SR = 16000
N_FFT = 2048
HOP_LENGTH = 64
WIN_LENGTH = 2048
WINDOW = "hann"
FRAME_RATE = 250

YIN_FMIN = 40.0
YIN_FMAX = 2000.0
EPS = 1e-8

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LOUDNESS_GATE_THRESHOLD = 0.01
PITCH_SMOOTH_WIN = 5
LOUD_SMOOTH_WIN = 5

DEFAULT_NOISE_SCALE = 0.03
PITCH_LOUDNESS_ONLY_NOISE_SCALE = 0.03
UNVOICED_FILL_MODE = "nearest"   # nearest | median

# Compact-control estimation from original audio
NUM_HARMONICS = 100
NUM_NOISE_BANDS = 64
HARMONIC_SAMPLE_BINS = 1
HARMONIC_ENERGY_SMOOTH_WIN = 5
NOISE_ENERGY_SMOOTH_WIN = 7

# =========================================================
# CURVE INTERPRETATION
# =========================================================
# direct_features:
#   y_curve_px      -> loudness_for_model
#   displacement_px -> pitch_hz
CONTROL_INPUT_MODE = "direct_features"
# choices:
#   "direct_features"
#   "curve_pixels"

USE_X_AS_PROGRESS = True
INVERT_LOUDNESS = False
INVERT_PITCH = False

TARGET_FRAME_MODE = "use_curve_rows"
# choices:
#   "use_curve_rows"
#   "match_audio_reference"

FORCE_TARGET_NUM_FRAMES = None

# Direct-feature mode settings
CLIP_DIRECT_LOUDNESS_TO_01 = True
MIN_VALID_PITCH_HZ = 40.0
MAX_VALID_PITCH_HZ = 2000.0

APPLY_LOUDNESS_SMOOTHING = False
APPLY_PITCH_SMOOTHING = False

# Curve-pixel mode settings
CURVE_LOUDNESS_SMOOTH_WIN = 9
CURVE_PITCH_SMOOTH_WIN = 7
PITCH_MIN_HZ_FALLBACK = 55.0
PITCH_MAX_HZ_FALLBACK = 880.0

# =========================================================
# OPTIONAL MUSICAL QUANTIZATION
# =========================================================
USE_QUANTIZATION = False
QUANTIZER_MODE = "minor_pentatonic"
QUANTIZER_ROOT_PC = 0
USE_STICKY_QUANTIZER = True
STICK_THRESHOLD_SEMITONES = 0.45
PREV_NOTE_PENALTY = 0.20

USE_NOTE_HOLD = False
MIN_NOTE_HOLD_FRAMES = 20

# =========================================================
# FORMULA-BASED GENERATION
# =========================================================
AMPLITUDE_EXPONENT = 1.5
AMPLITUDE_SMOOTH_WIN = 7

BASE_HARMONIC_DECAY = 1.35
BRIGHTNESS_FROM_CURVE = True
BRIGHTNESS_FROM_PITCH = True
CURVE_BRIGHTNESS_WEIGHT = 0.50
PITCH_BRIGHTNESS_WEIGHT = 0.50

USE_ATTACK_FROM_SLOPE = True
ATTACK_BOOST = 0.25

NOISE_BASE = 0.01
NOISE_FROM_LOUDNESS = 0.02
NOISE_FROM_SLOPE = 0.03
NOISE_SMOOTH_WIN = 7

# =========================================================
# AUDIO-BASED GENERATION
# =========================================================
AUDIO_BASED_USE_DIRECT_AMPLITUDE = False
AUDIO_BASED_USE_DIRECT_HARMONICS = True
AUDIO_BASED_USE_DIRECT_NOISE = True
AUDIO_BASED_MODULATE_AMPLITUDE_BY_INPUT = True

# =========================================================
# RECONSTRUCTION
# =========================================================
ADD_NOISE_TO_AUDIO = False

# =========================================================
# DATA CLASSES
# =========================================================
@dataclass
class AnalysisMeta:
    sr: int
    duration_sec: float
    num_samples: int
    n_fft: int
    hop_length: int
    win_length: int
    window: str
    n_frames: int
    n_freq_bins: int
    frame_rate: int
    source_wav: str
    ckpt_path: str


# =========================================================
# BASIC UTILS
# =========================================================
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def maybe_round_df(df: pd.DataFrame) -> pd.DataFrame:
    if not ROUND_FLOATS_FOR_CSV:
        return df
    out = df.copy()
    float_cols = out.select_dtypes(include=["float32", "float64"]).columns
    out[float_cols] = out[float_cols].round(CSV_FLOAT_DECIMALS)
    return out


def moving_average(x: np.ndarray, win: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if win <= 1:
        return x.copy()

    win = int(win)
    if win % 2 == 0:
        win += 1

    pad = win // 2
    x_pad = np.pad(x, (pad, pad), mode="edge")
    kernel = np.ones(win, dtype=np.float32) / win
    return np.convolve(x_pad, kernel, mode="valid").astype(np.float32)


def median_filter_1d(x: np.ndarray, win: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if win <= 1:
        return x.copy()

    win = int(win)
    if win % 2 == 0:
        win += 1

    pad = win // 2
    x_pad = np.pad(x, (pad, pad), mode="edge")
    out = np.empty_like(x)
    for i in range(len(x)):
        out[i] = np.median(x_pad[i:i + win])
    return out.astype(np.float32)


def normalize_01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    xmin = float(np.min(x))
    xmax = float(np.max(x))
    if xmax <= xmin:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - xmin) / (xmax - xmin)).astype(np.float32)


def hz_to_midi(freq_hz: np.ndarray) -> np.ndarray:
    freq_hz = np.asarray(freq_hz, dtype=np.float32)
    out = np.zeros_like(freq_hz, dtype=np.float32)
    mask = freq_hz > 0
    out[mask] = 69.0 + 12.0 * np.log2(np.maximum(freq_hz[mask], 1e-6) / 440.0)
    return out.astype(np.float32)


def midi_to_hz(midi: np.ndarray) -> np.ndarray:
    midi = np.asarray(midi, dtype=np.float32)
    return (440.0 * (2.0 ** ((midi - 69.0) / 12.0))).astype(np.float32)


def load_audio_mono(wav_path: Path, sr: int = SR) -> np.ndarray:
    y, _ = librosa.load(str(wav_path), sr=sr, mono=True)
    return y.astype(np.float32, copy=False)


def load_audio_model(ckpt_path: Path) -> AudioSynthTrainer:
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing model checkpoint: {ckpt_path}")
    model = AudioSynthTrainer.load_from_checkpoint(
        str(ckpt_path),
        map_location=torch.device("cpu"),
    )
    model.eval()
    return model


def fill_unvoiced_pitch(pitch_hz: np.ndarray, mode: str = "nearest") -> np.ndarray:
    pitch_hz = np.asarray(pitch_hz, dtype=np.float32).copy()
    voiced = pitch_hz > 0

    if not np.any(voiced):
        raise ValueError("All pitch frames are unvoiced.")

    if mode == "median":
        fill = float(np.median(pitch_hz[voiced]))
        pitch_hz[~voiced] = fill
        return pitch_hz.astype(np.float32)

    idx = np.arange(len(pitch_hz))
    voiced_idx = idx[voiced]
    voiced_vals = pitch_hz[voiced]
    nearest = np.interp(idx, voiced_idx, voiced_vals).astype(np.float32)
    pitch_hz[~voiced] = nearest[~voiced]
    return pitch_hz.astype(np.float32)


def resize_control_width(arr: Optional[np.ndarray], target_width: int, fill_mode: str = "tile") -> Optional[np.ndarray]:
    if arr is None:
        return None
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("Expected 2D array.")
    n_frames, cur_width = arr.shape

    if cur_width == target_width:
        return arr.astype(np.float32)
    if cur_width > target_width:
        return arr[:, :target_width].astype(np.float32)

    if fill_mode == "tile":
        reps = int(np.ceil(target_width / float(cur_width)))
        out = np.tile(arr, (1, reps))[:, :target_width]
        return out.astype(np.float32)

    out = np.zeros((n_frames, target_width), dtype=np.float32)
    out[:, :cur_width] = arr
    return out


def resample_2d_rows(arr: np.ndarray, n_target: int) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    n_src = arr.shape[0]
    if n_src == n_target:
        return arr.copy()

    x_src = np.linspace(0.0, 1.0, n_src, dtype=np.float32)
    x_dst = np.linspace(0.0, 1.0, n_target, dtype=np.float32)

    out = np.empty((n_target, arr.shape[1]), dtype=np.float32)
    for j in range(arr.shape[1]):
        out[:, j] = np.interp(x_dst, x_src, arr[:, j]).astype(np.float32)
    return out


def sort_feature_cols(cols: List[str], prefix: str) -> List[str]:
    return sorted(
        [c for c in cols if c.startswith(prefix)],
        key=lambda x: int(x.split("_")[1])
    )


def make_feature_labels(prefix: str, n: int) -> List[str]:
    return [f"{prefix}_{i:03d}" for i in range(n)]


def enforce_dense_compact_layout(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rebuild harmonic/noise blocks so final columns are strictly:
      harmonic_000 ... harmonic_(n-1)
      noise_000 ... noise_(n-1)
    """
    df = df.copy()

    harmonic_cols_old = sort_feature_cols(list(df.columns), "harmonic_")
    noise_cols_old = sort_feature_cols(list(df.columns), "noise_")

    base_cols = [
        c for c in df.columns
        if not c.startswith("harmonic_") and not c.startswith("noise_")
    ]

    pieces = [df[base_cols].reset_index(drop=True)]

    if harmonic_cols_old:
        harm = df[harmonic_cols_old].to_numpy(dtype=np.float32, copy=True)
        harmonic_cols_new = make_feature_labels("harmonic", harm.shape[1])
        pieces.append(pd.DataFrame(harm, columns=harmonic_cols_new))

    if noise_cols_old:
        nz = df[noise_cols_old].to_numpy(dtype=np.float32, copy=True)
        noise_cols_new = make_feature_labels("noise", nz.shape[1])
        pieces.append(pd.DataFrame(nz, columns=noise_cols_new))

    out = pd.concat(pieces, axis=1)
    return out


def resample_curve_to_n_frames(x_src: np.ndarray, y_src: np.ndarray, n_target: int) -> np.ndarray:
    x_src = np.asarray(x_src, dtype=np.float32)
    y_src = np.asarray(y_src, dtype=np.float32)

    valid = np.isfinite(x_src) & np.isfinite(y_src)
    x_src = x_src[valid]
    y_src = y_src[valid]

    if len(x_src) < 2:
        raise ValueError("Need at least 2 valid curve points.")

    if USE_X_AS_PROGRESS:
        x_norm = (x_src - x_src[0]) / max(float(x_src[-1] - x_src[0]), EPS)
    else:
        x_norm = np.linspace(0.0, 1.0, len(y_src), dtype=np.float32)

    x_norm, uniq_idx = np.unique(x_norm, return_index=True)
    y_src = y_src[uniq_idx]

    x_dst = np.linspace(0.0, 1.0, n_target, dtype=np.float32)
    y_dst = np.interp(x_dst, x_norm, y_src).astype(np.float32)
    return y_dst


def get_scale_pitch_classes(mode: str) -> np.ndarray:
    scale_map = {
        "minor_pentatonic": np.array([0, 3, 5, 7, 10], dtype=np.float32),
        "major_pentatonic": np.array([0, 2, 4, 7, 9], dtype=np.float32),
        "natural_minor":    np.array([0, 2, 3, 5, 7, 8, 10], dtype=np.float32),
        "major":            np.array([0, 2, 4, 5, 7, 9, 11], dtype=np.float32),
        "chromatic":        np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], dtype=np.float32),
        "blues_minor":      np.array([0, 3, 5, 6, 7, 10], dtype=np.float32),
    }
    if mode not in scale_map:
        raise ValueError(f"Unknown quantizer mode: {mode}")
    return scale_map[mode]


def quantize_to_scale(
    freq_hz: np.ndarray,
    mode: str = "minor_pentatonic",
    root_pc: int = 0,
    sticky: bool = True,
    stick_threshold: float = 0.45,
    prev_penalty: float = 0.20,
) -> np.ndarray:
    freq_hz = np.asarray(freq_hz, dtype=np.float32)
    midi = hz_to_midi(freq_hz)

    allowed_pc = (get_scale_pitch_classes(mode) + float(root_pc)) % 12.0
    out_midi = np.zeros_like(midi, dtype=np.float32)

    prev_q = None
    for i, m in enumerate(midi):
        if freq_hz[i] <= 0:
            out_midi[i] = 0.0
            continue

        octave = np.floor(m / 12.0)
        candidates = np.concatenate([
            allowed_pc + 12.0 * (octave - 1.0),
            allowed_pc + 12.0 * octave,
            allowed_pc + 12.0 * (octave + 1.0),
        ])

        if (not sticky) or (prev_q is None):
            chosen = candidates[np.argmin(np.abs(candidates - m))]
        else:
            if abs(m - prev_q) <= stick_threshold:
                chosen = prev_q
            else:
                costs = np.abs(candidates - m) + prev_penalty * np.abs(candidates - prev_q)
                chosen = candidates[np.argmin(costs)]

        out_midi[i] = float(chosen)
        prev_q = float(chosen)

    out_hz = midi_to_hz(out_midi)
    out_hz[freq_hz <= 0] = 0.0
    return out_hz.astype(np.float32)


def hold_notes(freq_hz: np.ndarray, min_hold_frames: int) -> np.ndarray:
    x = np.asarray(freq_hz, dtype=np.float32).copy()
    if len(x) == 0 or min_hold_frames <= 1:
        return x

    midi = hz_to_midi(x)
    voiced = x > 0

    i = 0
    n = len(x)
    while i < n:
        if not voiced[i]:
            i += 1
            continue

        start = i
        ref = midi[i]
        j = i + 1
        while j < n and voiced[j] and abs(float(midi[j] - ref)) < 0.6:
            j += 1

        if (j - start) < min_hold_frames:
            end = min(n, start + min_hold_frames)
            mask = voiced[start:end]
            if np.any(mask):
                med = float(np.median(midi[start:end][mask]))
                midi[start:end] = med

        i = j

    out = midi_to_hz(midi)
    out[~voiced] = 0.0
    return out.astype(np.float32)


# =========================================================
# ANALYSIS EXTRACTION FROM INPUT AUDIO
# =========================================================
def compute_framewise_attributes(
    y: np.ndarray,
    input_wav: Path,
    ckpt_path: Path,
    sr: int = SR,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], AnalysisMeta]:
    duration = float(librosa.get_duration(y=y, sr=sr))
    num_samples = int(len(y))

    s_complex = librosa.stft(
        y,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        window=WINDOW,
        center=True,
    )
    s_mag = np.abs(s_complex).astype(np.float32, copy=False)

    n_frames = s_mag.shape[1]
    n_freq_bins = s_mag.shape[0]

    t_frames = librosa.times_like(
        np.empty(n_frames, dtype=np.float32),
        sr=sr,
        hop_length=HOP_LENGTH,
        n_fft=None,
    ).astype(np.float32, copy=False)

    rms = librosa.feature.rms(
        y=y,
        frame_length=WIN_LENGTH,
        hop_length=HOP_LENGTH,
        center=True,
    )[0].astype(np.float32, copy=False)

    rms_norm = rms / (float(np.max(rms)) + EPS)
    rms_norm = moving_average(rms_norm, LOUD_SMOOTH_WIN)
    rms_norm = np.clip(rms_norm, 0.0, 1.0)

    f0 = librosa.yin(
        y,
        fmin=YIN_FMIN,
        fmax=YIN_FMAX,
        sr=sr,
        frame_length=WIN_LENGTH,
        hop_length=HOP_LENGTH,
    )
    f0 = np.nan_to_num(f0, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

    voiced = ((f0 >= YIN_FMIN) & (f0 <= YIN_FMAX) & (rms_norm > LOUDNESS_GATE_THRESHOLD)).astype(np.int32)
    f0_clean = f0.copy()
    f0_clean[voiced == 0] = 0.0

    f0_smooth = median_filter_1d(f0_clean, PITCH_SMOOTH_WIN)
    f0_clean[f0_clean > 0] = f0_smooth[f0_clean > 0]

    zcr = librosa.feature.zero_crossing_rate(
        y,
        frame_length=WIN_LENGTH,
        hop_length=HOP_LENGTH,
        center=True,
    )[0].astype(np.float32, copy=False)

    spectral_centroid = librosa.feature.spectral_centroid(
        S=s_mag,
        sr=sr,
    )[0].astype(np.float32, copy=False)

    spectral_rolloff = librosa.feature.spectral_rolloff(
        S=s_mag,
        sr=sr,
        roll_percent=0.85,
    )[0].astype(np.float32, copy=False)

    spectral_flatness = librosa.feature.spectral_flatness(
        S=s_mag + EPS
    )[0].astype(np.float32, copy=False)

    n = min(
        len(t_frames),
        len(rms),
        len(rms_norm),
        len(f0_clean),
        len(voiced),
        len(zcr),
        len(spectral_centroid),
        len(spectral_rolloff),
        len(spectral_flatness),
    )

    attrs = {
        "frame_idx": np.arange(n, dtype=np.int32),
        "time_sec": t_frames[:n],
        "rms": rms[:n],
        "rms_norm": rms_norm[:n],
        "f0_hz": f0_clean[:n],
        "voiced": voiced[:n],
        "zcr": zcr[:n],
        "spectral_centroid_hz": spectral_centroid[:n],
        "spectral_rolloff_hz": spectral_rolloff[:n],
        "spectral_flatness": spectral_flatness[:n],
    }

    meta = AnalysisMeta(
        sr=sr,
        duration_sec=duration,
        num_samples=num_samples,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        window=WINDOW,
        n_frames=int(n),
        n_freq_bins=int(n_freq_bins),
        frame_rate=FRAME_RATE,
        source_wav=str(input_wav),
        ckpt_path=str(ckpt_path),
    )

    return s_complex[:, :n], attrs, meta


def estimate_compact_controls_from_audio(
    s_complex: np.ndarray,
    attrs: Dict[str, np.ndarray],
    meta: AnalysisMeta,
    num_harmonics: int = NUM_HARMONICS,
    num_noise_bands: int = NUM_NOISE_BANDS,
) -> Dict[str, np.ndarray]:
    pitch_hz = np.asarray(attrs["f0_hz"], dtype=np.float32)
    loudness = np.asarray(attrs["rms_norm"], dtype=np.float32)
    pitch_hz_for_model = fill_unvoiced_pitch(pitch_hz, mode=UNVOICED_FILL_MODE)

    mag = np.abs(s_complex).T.astype(np.float32)
    freqs = librosa.fft_frequencies(sr=meta.sr, n_fft=meta.n_fft).astype(np.float32)
    nyquist = meta.sr / 2.0

    n_frames = mag.shape[0]
    harmonic_mags = np.zeros((n_frames, num_harmonics), dtype=np.float32)
    harmonic_energy = np.zeros(n_frames, dtype=np.float32)
    total_energy = np.sum(mag, axis=1).astype(np.float32)

    for t in range(n_frames):
        f0 = float(pitch_hz[t])
        if f0 <= 0 or loudness[t] <= LOUDNESS_GATE_THRESHOLD:
            harmonic_mags[t, 0] = 1.0
            continue

        frame_mag = mag[t]

        for h in range(num_harmonics):
            k = h + 1
            fk = k * f0
            if fk >= nyquist:
                break

            bin_idx = int(np.argmin(np.abs(freqs - fk)))
            left = max(0, bin_idx - HARMONIC_SAMPLE_BINS)
            right = min(len(freqs), bin_idx + HARMONIC_SAMPLE_BINS + 1)
            harmonic_mags[t, h] = float(np.max(frame_mag[left:right]))

        harmonic_energy[t] = float(np.sum(harmonic_mags[t]))

    harmonics = harmonic_mags.copy()
    harmonics /= (np.sum(harmonics, axis=1, keepdims=True) + EPS)

    harmonic_energy_norm = normalize_01(harmonic_energy)
    harmonic_energy_norm = moving_average(harmonic_energy_norm, HARMONIC_ENERGY_SMOOTH_WIN)
    amplitudes = (0.65 * loudness + 0.35 * harmonic_energy_norm).astype(np.float32)
    amplitudes = np.clip(amplitudes, 0.0, 1.0)

    residual_energy = np.maximum(total_energy - harmonic_energy, 0.0)
    residual_ratio = residual_energy / (total_energy + EPS)
    residual_ratio = moving_average(residual_ratio.astype(np.float32), NOISE_ENERGY_SMOOTH_WIN)
    residual_ratio = np.clip(residual_ratio, 0.0, 1.0)

    noise_env = (residual_ratio * loudness).astype(np.float32)
    noise = np.tile(noise_env[:, None], (1, num_noise_bands)).astype(np.float32)

    quiet = loudness <= LOUDNESS_GATE_THRESHOLD
    amplitudes[quiet] = 0.0
    harmonics[quiet] = 0.0
    harmonics[quiet, 0] = 1.0
    noise[quiet] = 0.0

    return {
        "pitch_hz_for_model": pitch_hz_for_model.astype(np.float32),
        "loudness_for_model": loudness.astype(np.float32),
        "amplitudes": amplitudes.reshape(-1, 1).astype(np.float32),
        "harmonics": harmonics.astype(np.float32),
        "noise": noise.astype(np.float32),
    }


# =========================================================
# CURVE -> TARGET CONTROL BUILDING
# =========================================================
def load_curve_csv(curve_csv_path: Path) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    if not curve_csv_path.exists():
        raise FileNotFoundError(f"Missing curve CSV: {curve_csv_path}")

    df = pd.read_csv(curve_csv_path)
    required = {"x_px", "y_curve_px", "displacement_px"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Curve CSV missing columns: {missing}")

    df = df.sort_values("x_px").reset_index(drop=True)

    x = df["x_px"].to_numpy(dtype=np.float32)
    loud_src = df["y_curve_px"].to_numpy(dtype=np.float32)
    pitch_src = df["displacement_px"].to_numpy(dtype=np.float32)

    if INVERT_LOUDNESS:
        loud_src = -loud_src
    if INVERT_PITCH:
        pitch_src = -pitch_src

    return df, x, loud_src, pitch_src


def resolve_target_num_frames(curve_df: pd.DataFrame, audio_meta: AnalysisMeta) -> int:
    if FORCE_TARGET_NUM_FRAMES is not None:
        return int(FORCE_TARGET_NUM_FRAMES)

    if TARGET_FRAME_MODE == "use_curve_rows":
        return int(len(curve_df))

    if TARGET_FRAME_MODE == "match_audio_reference":
        return int(audio_meta.n_frames)

    raise ValueError(f"Unsupported TARGET_FRAME_MODE: {TARGET_FRAME_MODE}")


def build_curve_controls_direct_features(
    curve_df: pd.DataFrame,
    curve_x: np.ndarray,
    curve_loud: np.ndarray,
    curve_pitch: np.ndarray,
    target_n_frames: int,
) -> Tuple[pd.DataFrame, AnalysisMeta]:
    if len(curve_df) == target_n_frames:
        loudness = curve_loud.astype(np.float32).copy()
        pitch_hz = curve_pitch.astype(np.float32).copy()
        loud_src_resampled = curve_loud.astype(np.float32).copy()
        pitch_src_resampled = curve_pitch.astype(np.float32).copy()
    else:
        loud_src_resampled = resample_curve_to_n_frames(curve_x, curve_loud, target_n_frames)
        pitch_src_resampled = resample_curve_to_n_frames(curve_x, curve_pitch, target_n_frames)
        loudness = loud_src_resampled.astype(np.float32).copy()
        pitch_hz = pitch_src_resampled.astype(np.float32).copy()

    loudness = np.nan_to_num(loudness, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    if CLIP_DIRECT_LOUDNESS_TO_01:
        loudness = np.clip(loudness, 0.0, 1.0)

    if APPLY_LOUDNESS_SMOOTHING:
        loudness = moving_average(loudness, LOUD_SMOOTH_WIN)
        if CLIP_DIRECT_LOUDNESS_TO_01:
            loudness = np.clip(loudness, 0.0, 1.0)

    pitch_hz = np.nan_to_num(pitch_hz, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    invalid = (pitch_hz < MIN_VALID_PITCH_HZ) | (pitch_hz > MAX_VALID_PITCH_HZ)
    pitch_hz[invalid] = 0.0

    if APPLY_PITCH_SMOOTHING:
        voiced = pitch_hz > 0
        pitch_sm = median_filter_1d(pitch_hz, PITCH_SMOOTH_WIN)
        pitch_hz[voiced] = pitch_sm[voiced]

    quiet = loudness < LOUDNESS_GATE_THRESHOLD
    loudness[quiet] = 0.0
    pitch_hz[quiet] = 0.0

    if USE_QUANTIZATION:
        pitch_hz = quantize_to_scale(
            pitch_hz,
            mode=QUANTIZER_MODE,
            root_pc=QUANTIZER_ROOT_PC,
            sticky=USE_STICKY_QUANTIZER,
            stick_threshold=STICK_THRESHOLD_SEMITONES,
            prev_penalty=PREV_NOTE_PENALTY,
        )
        pitch_hz[quiet] = 0.0

    if USE_NOTE_HOLD:
        pitch_hz = hold_notes(pitch_hz, MIN_NOTE_HOLD_FRAMES)
        pitch_hz[quiet] = 0.0

    pitch_hz_for_model = pitch_hz.copy()
    if np.any(pitch_hz_for_model > 0):
        pitch_hz_for_model = fill_unvoiced_pitch(pitch_hz_for_model, mode=UNVOICED_FILL_MODE)
    else:
        pitch_hz_for_model = np.full((target_n_frames,), 220.0, dtype=np.float32)

    time_sec = np.arange(target_n_frames, dtype=np.float32) / float(FRAME_RATE)

    controls_df = pd.DataFrame({
        "frame_idx": np.arange(target_n_frames, dtype=np.int32),
        "time_sec": time_sec.astype(np.float32),
        "sr": np.full(target_n_frames, SR, dtype=np.int32),
        "hop_length": np.full(target_n_frames, HOP_LENGTH, dtype=np.int32),
        "curve_loud_src_resampled": loud_src_resampled.astype(np.float32),
        "curve_pitch_src_resampled": pitch_src_resampled.astype(np.float32),
        "pitch_hz_raw": pitch_hz.astype(np.float32),
        "pitch_hz_for_model": pitch_hz_for_model.astype(np.float32),
        "loudness_for_model": loudness.astype(np.float32),
    })

    duration_sec = float(target_n_frames) / float(FRAME_RATE)
    num_samples = int(target_n_frames * HOP_LENGTH)

    meta = AnalysisMeta(
        sr=SR,
        duration_sec=duration_sec,
        num_samples=num_samples,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        window=WINDOW,
        n_frames=target_n_frames,
        n_freq_bins=(N_FFT // 2 + 1),
        frame_rate=FRAME_RATE,
        source_wav=str(CURVE_CSV_PATH),
        ckpt_path=str(CKPT_PATH),
    )

    return controls_df, meta


def build_curve_controls_curve_pixels(
    curve_x: np.ndarray,
    curve_loud: np.ndarray,
    curve_pitch: np.ndarray,
    target_n_frames: int,
) -> Tuple[pd.DataFrame, AnalysisMeta]:
    loud_resampled = resample_curve_to_n_frames(curve_x, curve_loud, target_n_frames)
    pitch_resampled = resample_curve_to_n_frames(curve_x, curve_pitch, target_n_frames)

    loud_norm = normalize_01(loud_resampled)
    loud_norm = moving_average(loud_norm, CURVE_LOUDNESS_SMOOTH_WIN)
    loud_norm = np.clip(loud_norm, 0.0, 1.0)

    pitch_norm = normalize_01(pitch_resampled)
    pitch_norm = moving_average(pitch_norm, CURVE_PITCH_SMOOTH_WIN)
    pitch_norm = np.clip(pitch_norm, 0.0, 1.0)

    loudness = loud_norm.astype(np.float32)

    pitch_hz = PITCH_MIN_HZ_FALLBACK + pitch_norm * (PITCH_MAX_HZ_FALLBACK - PITCH_MIN_HZ_FALLBACK)
    pitch_hz = pitch_hz.astype(np.float32)

    quiet = loudness < LOUDNESS_GATE_THRESHOLD
    loudness[quiet] = 0.0
    pitch_hz[quiet] = 0.0

    voiced = pitch_hz > 0
    if np.any(voiced):
        pitch_sm = median_filter_1d(pitch_hz, CURVE_PITCH_SMOOTH_WIN)
        pitch_hz[voiced] = pitch_sm[voiced]

    if USE_QUANTIZATION:
        pitch_hz = quantize_to_scale(
            pitch_hz,
            mode=QUANTIZER_MODE,
            root_pc=QUANTIZER_ROOT_PC,
            sticky=USE_STICKY_QUANTIZER,
            stick_threshold=STICK_THRESHOLD_SEMITONES,
            prev_penalty=PREV_NOTE_PENALTY,
        )
        pitch_hz[quiet] = 0.0

    if USE_NOTE_HOLD:
        pitch_hz = hold_notes(pitch_hz, MIN_NOTE_HOLD_FRAMES)
        pitch_hz[quiet] = 0.0

    pitch_hz_for_model = pitch_hz.copy()
    if np.any(pitch_hz_for_model > 0):
        pitch_hz_for_model = fill_unvoiced_pitch(pitch_hz_for_model, mode=UNVOICED_FILL_MODE)
    else:
        pitch_hz_for_model = np.full((target_n_frames,), 220.0, dtype=np.float32)

    time_sec = np.arange(target_n_frames, dtype=np.float32) / float(FRAME_RATE)

    controls_df = pd.DataFrame({
        "frame_idx": np.arange(target_n_frames, dtype=np.int32),
        "time_sec": time_sec.astype(np.float32),
        "sr": np.full(target_n_frames, SR, dtype=np.int32),
        "hop_length": np.full(target_n_frames, HOP_LENGTH, dtype=np.int32),
        "curve_loud_src_resampled": loud_resampled.astype(np.float32),
        "curve_pitch_src_resampled": pitch_resampled.astype(np.float32),
        "curve_loud_norm": loud_norm.astype(np.float32),
        "curve_pitch_norm": pitch_norm.astype(np.float32),
        "pitch_hz_raw": pitch_hz.astype(np.float32),
        "pitch_hz_for_model": pitch_hz_for_model.astype(np.float32),
        "loudness_for_model": loudness.astype(np.float32),
    })

    duration_sec = float(target_n_frames) / float(FRAME_RATE)
    num_samples = int(target_n_frames * HOP_LENGTH)

    meta = AnalysisMeta(
        sr=SR,
        duration_sec=duration_sec,
        num_samples=num_samples,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        window=WINDOW,
        n_frames=target_n_frames,
        n_freq_bins=(N_FFT // 2 + 1),
        frame_rate=FRAME_RATE,
        source_wav=str(CURVE_CSV_PATH),
        ckpt_path=str(CKPT_PATH),
    )

    return controls_df, meta


def build_pitch_loudness_dataframe_from_curve_controls(
    controls_df: pd.DataFrame,
    meta: AnalysisMeta,
) -> pd.DataFrame:
    n_frames = len(controls_df)
    return pd.DataFrame({
        "frame_idx": np.arange(n_frames, dtype=np.int32),
        "sr": np.full(n_frames, meta.sr, dtype=np.int32),
        "hop_length": np.full(n_frames, meta.hop_length, dtype=np.int32),
        "pitch_hz_for_model": controls_df["pitch_hz_for_model"].to_numpy(dtype=np.float32),
        "loudness_for_model": controls_df["loudness_for_model"].to_numpy(dtype=np.float32),
    })


# =========================================================
# THREE GENERATION LOGICS
# =========================================================
def pitch_brightness_curve(pitch_hz: np.ndarray) -> np.ndarray:
    pitch_hz = np.asarray(pitch_hz, dtype=np.float32)
    voiced = pitch_hz > 0
    out = np.zeros_like(pitch_hz, dtype=np.float32)
    if np.any(voiced):
        midi = hz_to_midi(pitch_hz[voiced])
        out[voiced] = ((midi - np.min(midi)) / max(float(np.max(midi) - np.min(midi)), EPS)).astype(np.float32)
    return out


def build_formula_based_controls(
    controls_df: pd.DataFrame,
    num_harmonics: int,
    num_noise_bands: int,
) -> Dict[str, np.ndarray]:
    pitch_hz = controls_df["pitch_hz_raw"].to_numpy(dtype=np.float32)
    pitch_hz_for_model = controls_df["pitch_hz_for_model"].to_numpy(dtype=np.float32)
    loudness = controls_df["loudness_for_model"].to_numpy(dtype=np.float32)

    if "curve_loud_norm" in controls_df.columns:
        slope_src = controls_df["curve_loud_norm"].to_numpy(dtype=np.float32)
    else:
        slope_src = loudness.copy()

    n_frames = len(controls_df)
    harmonics = np.zeros((n_frames, num_harmonics), dtype=np.float32)
    amplitudes = np.zeros((n_frames, 1), dtype=np.float32)
    noise = np.zeros((n_frames, num_noise_bands), dtype=np.float32) if num_noise_bands > 0 else None

    slope = np.diff(slope_src, prepend=slope_src[0]).astype(np.float32)
    slope_abs = normalize_01(np.abs(slope))
    pitch_bright = pitch_brightness_curve(pitch_hz_for_model)

    harmonic_idx = np.arange(1, num_harmonics + 1, dtype=np.float32)

    for t in range(n_frames):
        l = float(loudness[t])
        p = float(pitch_hz[t])

        if l <= LOUDNESS_GATE_THRESHOLD or p <= 0:
            harmonics[t, 0] = 1.0
            amplitudes[t, 0] = 0.0
            if noise is not None:
                noise[t, :] = 0.0
            continue

        brightness = 0.0
        if BRIGHTNESS_FROM_CURVE:
            brightness += CURVE_BRIGHTNESS_WEIGHT * float(l)
        if BRIGHTNESS_FROM_PITCH:
            brightness += PITCH_BRIGHTNESS_WEIGHT * float(pitch_bright[t])
        brightness = float(np.clip(brightness, 0.0, 1.0))

        local_decay = BASE_HARMONIC_DECAY + 0.8 * (1.0 - brightness)

        h = 1.0 / np.power(harmonic_idx, local_decay)
        h[0] *= 1.35
        if num_harmonics > 1:
            h[1] *= 1.15
        if num_harmonics > 2:
            h[2] *= 1.08

        if USE_ATTACK_FROM_SLOPE:
            attack = float(slope_abs[t])
            h[:min(8, num_harmonics)] *= (1.0 + ATTACK_BOOST * attack)

        h = h / (np.sum(h) + EPS)
        harmonics[t] = h.astype(np.float32)

        amp = np.power(np.clip(l, 0.0, 1.0), AMPLITUDE_EXPONENT)
        if USE_ATTACK_FROM_SLOPE:
            amp *= (1.0 + 0.20 * float(slope_abs[t]))
        amplitudes[t, 0] = float(amp)

        if noise is not None:
            noise_level = NOISE_BASE + NOISE_FROM_LOUDNESS * l + NOISE_FROM_SLOPE * float(slope_abs[t])
            noise[t, :] = float(noise_level)

    amplitudes[:, 0] = moving_average(amplitudes[:, 0], AMPLITUDE_SMOOTH_WIN)
    amplitudes[:, 0] = np.clip(amplitudes[:, 0], 0.0, 1.0)

    for t in range(n_frames):
        if loudness[t] <= LOUDNESS_GATE_THRESHOLD:
            harmonics[t] = 0.0
            harmonics[t, 0] = 1.0
            amplitudes[t, 0] = 0.0
            if noise is not None:
                noise[t] = 0.0

    if noise is not None:
        noise_env = noise[:, 0]
        noise_env = moving_average(noise_env, NOISE_SMOOTH_WIN)
        noise_env = np.clip(noise_env, 0.0, 1.0)
        noise = np.tile(noise_env[:, None], (1, num_noise_bands)).astype(np.float32)

    return {
        "pitch_hz_for_model": pitch_hz_for_model.astype(np.float32),
        "loudness_for_model": loudness.astype(np.float32),
        "amplitudes": amplitudes.astype(np.float32),
        "harmonics": harmonics.astype(np.float32),
        "noise": noise,
    }


def predict_synth_controls_from_pitch_loudness(
    model: AudioSynthTrainer,
    pitch_hz_for_model: np.ndarray,
    loudness_for_model: np.ndarray,
) -> Dict[str, np.ndarray]:
    p_t = torch.from_numpy(pitch_hz_for_model).float().reshape(1, -1, 1).to(DEVICE)
    l_t = torch.from_numpy(loudness_for_model).float().reshape(1, -1, 1).to(DEVICE)

    with torch.no_grad():
        controls = model.model(p_t, l_t)

    amplitudes = controls.get("amplitudes", None)
    harmonics = controls.get("harmonics", None)
    noise = controls.get("noise", None)

    if amplitudes is None:
        raise KeyError("Model output missing 'amplitudes'")
    if harmonics is None:
        raise KeyError("Model output missing 'harmonics'")

    return {
        "pitch_hz_for_model": pitch_hz_for_model.astype(np.float32),
        "loudness_for_model": loudness_for_model.astype(np.float32),
        "amplitudes": amplitudes.squeeze(0).detach().cpu().numpy().astype(np.float32),
        "harmonics": harmonics.squeeze(0).detach().cpu().numpy().astype(np.float32),
        "noise": None if noise is None else noise.squeeze(0).detach().cpu().numpy().astype(np.float32),
    }


def build_audio_based_controls(
    controls_df: pd.DataFrame,
    audio_reference_controls: Dict[str, np.ndarray],
    target_num_harmonics: int,
    target_num_noise_bands: int,
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

    if AUDIO_BASED_USE_DIRECT_AMPLITUDE:
        amplitudes = ref_amp.astype(np.float32)
    else:
        amplitudes = np.power(np.clip(loudness_for_model, 0.0, 1.0), AMPLITUDE_EXPONENT).reshape(-1, 1).astype(np.float32)
        amplitudes[:, 0] = moving_average(amplitudes[:, 0], AMPLITUDE_SMOOTH_WIN)

    if AUDIO_BASED_MODULATE_AMPLITUDE_BY_INPUT:
        ref_amp_norm = ref_amp[:, 0].copy()
        if np.max(ref_amp_norm) > 0:
            ref_amp_norm = ref_amp_norm / (np.max(ref_amp_norm) + EPS)
        amplitudes[:, 0] *= (0.5 + 0.5 * ref_amp_norm)

    harmonics = ref_harm.astype(np.float32) if AUDIO_BASED_USE_DIRECT_HARMONICS else None
    noise = ref_noise.astype(np.float32) if (AUDIO_BASED_USE_DIRECT_NOISE and ref_noise is not None) else None

    if harmonics is None:
        rb = build_formula_based_controls(
            controls_df=controls_df,
            num_harmonics=target_num_harmonics,
            num_noise_bands=target_num_noise_bands,
        )
        harmonics = rb["harmonics"]
        if noise is None:
            noise = rb["noise"]

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
# DATAFRAME BUILDERS
# =========================================================
def build_compact_dataframe_from_controls(
    compact_controls: Dict[str, np.ndarray],
    meta: AnalysisMeta,
) -> pd.DataFrame:
    n_frames = len(compact_controls["pitch_hz_for_model"])

    base_df = pd.DataFrame({
        "frame_idx": np.arange(n_frames, dtype=np.int32),
        "sr": np.full(n_frames, meta.sr, dtype=np.int32),
        "hop_length": np.full(n_frames, meta.hop_length, dtype=np.int32),
        "pitch_hz_for_model": compact_controls["pitch_hz_for_model"].astype(np.float32),
        "loudness_for_model": compact_controls["loudness_for_model"].astype(np.float32),
        "amplitudes": compact_controls["amplitudes"].reshape(-1).astype(np.float32),
    })

    pieces = [base_df]

    harmonics = compact_controls["harmonics"].astype(np.float32)
    harmonic_cols = make_feature_labels("harmonic", harmonics.shape[1])
    pieces.append(pd.DataFrame(harmonics, columns=harmonic_cols))

    if compact_controls["noise"] is not None:
        noise = compact_controls["noise"].astype(np.float32)
        noise_cols = make_feature_labels("noise", noise.shape[1])
        pieces.append(pd.DataFrame(noise, columns=noise_cols))

    out = pd.concat(pieces, axis=1)
    out = enforce_dense_compact_layout(out)
    return out


def build_full_dataframe(
    s_complex: np.ndarray,
    attrs: Dict[str, np.ndarray],
    compact_controls: Dict[str, np.ndarray],
    meta: AnalysisMeta,
) -> pd.DataFrame:
    n_frames = len(attrs["frame_idx"])
    n_freq_bins = int(meta.n_freq_bins)

    base_df = pd.DataFrame({
        "frame_idx": attrs["frame_idx"].astype(np.int32),
        "time_sec": attrs["time_sec"].astype(np.float32),
        "sr": np.full(n_frames, meta.sr, dtype=np.int32),
        "num_samples": np.full(n_frames, meta.num_samples, dtype=np.int32),
        "n_fft": np.full(n_frames, meta.n_fft, dtype=np.int32),
        "hop_length": np.full(n_frames, meta.hop_length, dtype=np.int32),
        "win_length": np.full(n_frames, meta.win_length, dtype=np.int32),
        "n_frames_total": np.full(n_frames, meta.n_frames, dtype=np.int32),
        "n_freq_bins": np.full(n_frames, meta.n_freq_bins, dtype=np.int32),
        "frame_rate": np.full(n_frames, meta.frame_rate, dtype=np.int32),
        "source_wav": np.full(n_frames, meta.source_wav),
        "ckpt_path": np.full(n_frames, meta.ckpt_path),
        "rms": attrs["rms"].astype(np.float32),
        "rms_norm": attrs["rms_norm"].astype(np.float32),
        "f0_hz": attrs["f0_hz"].astype(np.float32),
        "voiced": attrs["voiced"].astype(np.int32),
        "zcr": attrs["zcr"].astype(np.float32),
        "spectral_centroid_hz": attrs["spectral_centroid_hz"].astype(np.float32),
        "spectral_rolloff_hz": attrs["spectral_rolloff_hz"].astype(np.float32),
        "spectral_flatness": attrs["spectral_flatness"].astype(np.float32),
        "pitch_hz_for_model": compact_controls["pitch_hz_for_model"].astype(np.float32),
        "loudness_for_model": compact_controls["loudness_for_model"].astype(np.float32),
        "amplitudes": compact_controls["amplitudes"].reshape(-1).astype(np.float32),
    })

    pieces = [base_df]

    harmonics = compact_controls["harmonics"].astype(np.float32)
    harmonic_cols = make_feature_labels("harmonic", harmonics.shape[1])
    pieces.append(pd.DataFrame(harmonics, columns=harmonic_cols))

    if compact_controls["noise"] is not None:
        noise = compact_controls["noise"].astype(np.float32)
        noise_cols = make_feature_labels("noise", noise.shape[1])
        pieces.append(pd.DataFrame(noise, columns=noise_cols))

    stft_real = np.real(s_complex).T.astype(np.float32)
    stft_imag = np.imag(s_complex).T.astype(np.float32)

    real_cols = [f"stft_real_bin_{b:04d}" for b in range(n_freq_bins)]
    imag_cols = [f"stft_imag_bin_{b:04d}" for b in range(n_freq_bins)]

    pieces.append(pd.DataFrame(stft_real, columns=real_cols))
    pieces.append(pd.DataFrame(stft_imag, columns=imag_cols))

    out = pd.concat(pieces, axis=1)
    out = enforce_dense_compact_layout(out)
    return out


# =========================================================
# RECONSTRUCTION
# =========================================================
def synthesize_from_controls(
    pitch_hz_for_model: np.ndarray,
    loudness_for_model: np.ndarray,
    amplitudes: np.ndarray,
    harmonics: np.ndarray,
    noise: Optional[np.ndarray],
    sr: int,
    hop_length: int,
    out_wav: Path,
    noise_scale: float = DEFAULT_NOISE_SCALE,
) -> Tuple[np.ndarray, int]:
    p_t = torch.from_numpy(pitch_hz_for_model).float().reshape(1, -1, 1).to(DEVICE)
    a_t = torch.from_numpy(amplitudes).float().reshape(1, -1, 1).to(DEVICE)
    h_t = torch.from_numpy(harmonics).float().reshape(1, harmonics.shape[0], harmonics.shape[1]).to(DEVICE)

    n_samples = p_t.shape[1] * hop_length

    with torch.no_grad():
        harm = harmonic_synthesis(
            p_t,
            a_t,
            h_t,
            n_samples=n_samples,
            sample_rate=sr,
        )

        if noise is not None and noise.shape[1] > 0:
            n_t = torch.from_numpy(noise).float().reshape(1, noise.shape[0], noise.shape[1]).to(DEVICE)
            nz = noise_synthesis(n_t * noise_scale, n_samples=n_samples)
            audio = harm + nz if ADD_NOISE_TO_AUDIO else harm
        else:
            audio = harm

    wav = audio.squeeze().detach().cpu().numpy().astype(np.float32)

    gate_audio = np.repeat((loudness_for_model > LOUDNESS_GATE_THRESHOLD).astype(np.float32), hop_length)
    if len(gate_audio) < len(wav):
        gate_audio = np.pad(gate_audio, (0, len(wav) - len(gate_audio)), mode="edge")
    else:
        gate_audio = gate_audio[:len(wav)]
    wav = wav * gate_audio

    peak = float(np.max(np.abs(wav)))
    if peak > 0:
        wav = wav / peak * 0.95

    sf.write(out_wav, wav, sr)
    return wav, sr


def reconstruct_from_compact_csv(
    df: pd.DataFrame,
    out_wav: Path,
    noise_scale: float = DEFAULT_NOISE_SCALE,
) -> Tuple[np.ndarray, int]:
    df = enforce_dense_compact_layout(df)
    df = df.sort_values("frame_idx").reset_index(drop=True)

    sr = int(df["sr"].iloc[0])
    hop_length = int(df["hop_length"].iloc[0])

    pitch_hz_for_model = df["pitch_hz_for_model"].to_numpy(dtype=np.float32)
    loudness_for_model = df["loudness_for_model"].to_numpy(dtype=np.float32)
    amplitudes = df["amplitudes"].to_numpy(dtype=np.float32).reshape(-1, 1)

    harmonic_cols = sort_feature_cols(list(df.columns), "harmonic_")
    noise_cols = sort_feature_cols(list(df.columns), "noise_")

    if len(harmonic_cols) == 0:
        raise ValueError("Compact CSV is missing harmonic_* columns.")

    harmonics = df[harmonic_cols].to_numpy(dtype=np.float32)
    noise = df[noise_cols].to_numpy(dtype=np.float32) if len(noise_cols) > 0 else None

    return synthesize_from_controls(
        pitch_hz_for_model=pitch_hz_for_model,
        loudness_for_model=loudness_for_model,
        amplitudes=amplitudes,
        harmonics=harmonics,
        noise=noise,
        sr=sr,
        hop_length=hop_length,
        out_wav=out_wav,
        noise_scale=noise_scale,
    )


def reconstruct_from_pitch_loudness_csv(
    df: pd.DataFrame,
    ckpt_path: Path,
    out_wav: Path,
    noise_scale: float = PITCH_LOUDNESS_ONLY_NOISE_SCALE,
) -> Tuple[np.ndarray, int]:
    df = df.sort_values("frame_idx").reset_index(drop=True)

    sr = int(df["sr"].iloc[0])
    hop_length = int(df["hop_length"].iloc[0])

    pitch_hz_for_model = df["pitch_hz_for_model"].to_numpy(dtype=np.float32)
    loudness_for_model = df["loudness_for_model"].to_numpy(dtype=np.float32)

    model = load_audio_model(ckpt_path).to(DEVICE)
    synth_controls = predict_synth_controls_from_pitch_loudness(
        model=model,
        pitch_hz_for_model=pitch_hz_for_model,
        loudness_for_model=loudness_for_model,
    )

    synth_controls["harmonics"] = resize_control_width(
        synth_controls["harmonics"], NUM_HARMONICS, fill_mode="tile"
    )
    synth_controls["noise"] = (
        resize_control_width(synth_controls["noise"], NUM_NOISE_BANDS, fill_mode="tile")
        if (synth_controls["noise"] is not None and NUM_NOISE_BANDS > 0)
        else None
    )

    return synthesize_from_controls(
        pitch_hz_for_model=synth_controls["pitch_hz_for_model"],
        loudness_for_model=synth_controls["loudness_for_model"],
        amplitudes=synth_controls["amplitudes"].reshape(-1, 1),
        harmonics=synth_controls["harmonics"],
        noise=synth_controls["noise"],
        sr=sr,
        hop_length=hop_length,
        out_wav=out_wav,
        noise_scale=noise_scale,
    )


# =========================================================
# LONG VISUALIZATION / REPORTING
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


def build_all_modes_long_df(mode_to_compact: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    parts = []
    for mode_name, compact_df in mode_to_compact.items():
        parts.append(compact_to_long_timeseries(compact_df=compact_df, mode_name=mode_name))
    return pd.concat(parts, axis=0, ignore_index=True)


def plot_summary_controls(controls_df: pd.DataFrame, plot_path: Path) -> None:
    t = controls_df["time_sec"].to_numpy(dtype=np.float32)

    fig, ax = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    ax[0].plot(t, controls_df["loudness_for_model"].to_numpy(dtype=np.float32), linewidth=1.0)
    ax[0].set_title("Curve-derived loudness_for_model")
    ax[0].grid(alpha=0.3)

    ax[1].plot(t, controls_df["pitch_hz_raw"].to_numpy(dtype=np.float32), linewidth=1.0, label="pitch_hz_raw")
    ax[1].plot(t, controls_df["pitch_hz_for_model"].to_numpy(dtype=np.float32), linewidth=1.0, label="pitch_hz_for_model")
    ax[1].set_title("Curve-derived pitch")
    ax[1].grid(alpha=0.3)
    ax[1].legend()
    ax[1].set_xlabel("Time (sec)")

    fig.tight_layout()
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)


def plot_components_long_df(long_df: pd.DataFrame, plot_path: Path) -> None:
    mode_order = ["formula_based", "ml_based", "audio_based"]
    ordered_modes = [m for m in mode_order if m in set(long_df["mode"].unique())]

    fig, ax = plt.subplots(3, 1, figsize=(16, 14), sharex=True)

    for mode_name in ordered_modes:
        sub = long_df[(long_df["mode"] == mode_name) & (long_df["component_type"] == "pitch")].sort_values("frame_idx")
        ax[0].plot(sub["time_sec"], sub["value"], linewidth=1.0, label=mode_name)
    ax[0].set_title("Pitch time series")
    ax[0].grid(alpha=0.3)
    ax[0].legend()

    for mode_name in ordered_modes:
        sub = long_df[(long_df["mode"] == mode_name) & (long_df["component_type"] == "loudness")].sort_values("frame_idx")
        ax[1].plot(sub["time_sec"], sub["value"], linewidth=1.0, label=mode_name)
    ax[1].set_title("Loudness time series")
    ax[1].grid(alpha=0.3)
    ax[1].legend()

    for mode_name in ordered_modes:
        sub = long_df[(long_df["mode"] == mode_name) & (long_df["component_type"] == "amplitude")].sort_values("frame_idx")
        ax[2].plot(sub["time_sec"], sub["value"], linewidth=1.0, label=mode_name)
    ax[2].set_title("Amplitude time series")
    ax[2].grid(alpha=0.3)
    ax[2].legend()
    ax[2].set_xlabel("Time (sec)")

    fig.tight_layout()
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)


def save_metadata_json(meta_path: Path, payload: Dict[str, object]) -> None:
    meta_path.write_text(json.dumps(payload, indent=2))


# =========================================================
# PIPELINE
# =========================================================
def run_generate() -> None:
    ensure_dir(OUTPUT_DIR)

    print("=" * 100)
    print("SELF-SUFFICIENT CURVE -> CONTROLS -> COMPACT CSV -> AUDIO")
    print("=" * 100)
    print(f"Curve CSV                  : {CURVE_CSV_PATH}")
    print(f"Input audio WAV            : {INPUT_WAV}")
    print(f"Checkpoint                 : {CKPT_PATH}")
    print(f"Generation mode            : {GENERATION_MODE}")
    print(f"CONTROL_INPUT_MODE         : {CONTROL_INPUT_MODE}")
    print(f"TARGET_FRAME_MODE          : {TARGET_FRAME_MODE}")
    print(f"Output dir                 : {OUTPUT_DIR}")
    print(f"Device                     : {DEVICE}")

    print("\n[1/8] Loading curve CSV...")
    curve_df, x_curve, loud_curve, pitch_curve = load_curve_csv(CURVE_CSV_PATH)

    print("[2/8] Extracting internal audio-derived reference controls from input WAV...")
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
    audio_reference_full_df = build_full_dataframe(
        s_complex=s_complex,
        attrs=audio_attrs,
        compact_controls=audio_reference_controls,
        meta=audio_meta,
    )

    audio_reference_compact_df = maybe_round_df(audio_reference_compact_df)
    audio_reference_full_df = maybe_round_df(audio_reference_full_df)

    audio_reference_compact_df.to_csv(OUT_AUDIO_ANALYSIS_COMPACT_CSV, index=False)
    audio_reference_full_df.to_csv(OUT_AUDIO_ANALYSIS_FULL_CSV, index=False)

    print(f"Saved internal audio reference compact CSV : {OUT_AUDIO_ANALYSIS_COMPACT_CSV}")
    print(f"Saved internal audio reference full CSV    : {OUT_AUDIO_ANALYSIS_FULL_CSV}")

    print("[3/8] Resolving target frame count...")
    target_n_frames = resolve_target_num_frames(curve_df, audio_meta)
    print(f"Curve rows                 : {len(curve_df)}")
    print(f"Target frames              : {target_n_frames}")

    print("[4/8] Building curve-derived pitch/loudness controls...")
    if CONTROL_INPUT_MODE == "direct_features":
        curve_controls_df, curve_meta = build_curve_controls_direct_features(
            curve_df=curve_df,
            curve_x=x_curve,
            curve_loud=loud_curve,
            curve_pitch=pitch_curve,
            target_n_frames=target_n_frames,
        )
    elif CONTROL_INPUT_MODE == "curve_pixels":
        curve_controls_df, curve_meta = build_curve_controls_curve_pixels(
            curve_x=x_curve,
            curve_loud=loud_curve,
            curve_pitch=pitch_curve,
            target_n_frames=target_n_frames,
        )
    else:
        raise ValueError(f"Unknown CONTROL_INPUT_MODE: {CONTROL_INPUT_MODE}")

    pitch_loudness_df = build_pitch_loudness_dataframe_from_curve_controls(curve_controls_df, curve_meta)

    curve_controls_df = maybe_round_df(curve_controls_df)
    pitch_loudness_df = maybe_round_df(pitch_loudness_df)

    curve_controls_df.to_csv(OUT_CURVE_CONTROLS_CSV, index=False)
    pitch_loudness_df.to_csv(OUT_PITCH_LOUDNESS_CSV, index=False)

    print(f"Saved curve controls CSV   : {OUT_CURVE_CONTROLS_CSV}")
    print(f"Saved pitch+loudness CSV   : {OUT_PITCH_LOUDNESS_CSV}")

    print("[5/8] Generating compact controls...")
    if GENERATION_MODE == "all":
        modes = ["formula_based", "ml_based", "audio_based"]
    else:
        modes = [GENERATION_MODE]

    mode_to_compact_df: Dict[str, pd.DataFrame] = {}
    mode_to_csv_path: Dict[str, str] = {}
    mode_to_wav_path: Dict[str, str] = {}

    print("[6/8] Running selected modes...")
    model = None
    for mode_name in modes:
        print(f"  -> {mode_name}")

        if mode_name == "formula_based":
            compact_controls = build_formula_based_controls(
                controls_df=curve_controls_df,
                num_harmonics=NUM_HARMONICS,
                num_noise_bands=NUM_NOISE_BANDS,
            )

        elif mode_name == "ml_based":
            if model is None:
                model = load_audio_model(CKPT_PATH).to(DEVICE)
            compact_controls = predict_synth_controls_from_pitch_loudness(
                model=model,
                pitch_hz_for_model=curve_controls_df["pitch_hz_for_model"].to_numpy(dtype=np.float32),
                loudness_for_model=curve_controls_df["loudness_for_model"].to_numpy(dtype=np.float32),
            )
            compact_controls["harmonics"] = resize_control_width(
                compact_controls["harmonics"], NUM_HARMONICS, fill_mode="tile"
            )
            compact_controls["noise"] = (
                resize_control_width(compact_controls["noise"], NUM_NOISE_BANDS, fill_mode="tile")
                if (compact_controls["noise"] is not None and NUM_NOISE_BANDS > 0)
                else None
            )

        elif mode_name == "audio_based":
            compact_controls = build_audio_based_controls(
                controls_df=curve_controls_df,
                audio_reference_controls=audio_reference_controls,
                target_num_harmonics=NUM_HARMONICS,
                target_num_noise_bands=NUM_NOISE_BANDS,
            )

        else:
            raise ValueError(f"Unknown mode: {mode_name}")

        compact_df = build_compact_dataframe_from_controls(compact_controls, curve_meta)
        compact_df = enforce_dense_compact_layout(compact_df)
        compact_df = maybe_round_df(compact_df)

        out_csv = OUTPUT_DIR / f"{RUN_PREFIX}_{mode_name}.csv"
        out_wav = OUTPUT_DIR / f"{RUN_PREFIX}_{mode_name}.wav"

        compact_df.to_csv(out_csv, index=False)
        reconstruct_from_compact_csv(compact_df, out_wav, noise_scale=DEFAULT_NOISE_SCALE)

        mode_to_compact_df[mode_name] = compact_df
        mode_to_csv_path[mode_name] = str(out_csv)
        mode_to_wav_path[mode_name] = str(out_wav)

        print(f"     CSV: {out_csv}")
        print(f"     WAV: {out_wav}")

    print("[7/8] Building plots and long visualization CSV...")
    long_df = build_all_modes_long_df(mode_to_compact_df)
    long_df = maybe_round_df(long_df)
    long_df.to_csv(OUT_LONG_VIS_CSV, index=False)

    plot_summary_controls(curve_controls_df, OUT_SUMMARY_PLOT)
    plot_components_long_df(long_df, OUT_COMPONENTS_PLOT)

    print(f"Saved long visualization CSV : {OUT_LONG_VIS_CSV}")
    print(f"Saved summary plot           : {OUT_SUMMARY_PLOT}")
    print(f"Saved components plot        : {OUT_COMPONENTS_PLOT}")

    print("[8/8] Saving metadata...")
    meta_payload = {
        "run_mode": RUN_MODE,
        "generation_mode": GENERATION_MODE,
        "control_input_mode": CONTROL_INPUT_MODE,
        "target_frame_mode": TARGET_FRAME_MODE,
        "curve_csv_path": str(CURVE_CSV_PATH),
        "input_wav_path": str(INPUT_WAV),
        "checkpoint_path": str(CKPT_PATH),
        "curve_rows": int(len(curve_df)),
        "audio_reference_frames": int(audio_meta.n_frames),
        "target_frames": int(curve_meta.n_frames),
        "num_harmonics": int(NUM_HARMONICS),
        "num_noise_bands": int(NUM_NOISE_BANDS),
        "use_quantization": bool(USE_QUANTIZATION),
        "quantizer_mode": QUANTIZER_MODE,
        "quantizer_root_pc": QUANTIZER_ROOT_PC,
        "add_noise_to_audio": bool(ADD_NOISE_TO_AUDIO),
        "curve_controls_csv_path": str(OUT_CURVE_CONTROLS_CSV),
        "pitch_loudness_csv_path": str(OUT_PITCH_LOUDNESS_CSV),
        "audio_reference_compact_csv_path": str(OUT_AUDIO_ANALYSIS_COMPACT_CSV),
        "audio_reference_full_csv_path": str(OUT_AUDIO_ANALYSIS_FULL_CSV),
        "long_visualization_csv_path": str(OUT_LONG_VIS_CSV),
        "summary_plot_path": str(OUT_SUMMARY_PLOT),
        "components_plot_path": str(OUT_COMPONENTS_PLOT),
        "mode_csv_paths": mode_to_csv_path,
        "mode_wav_paths": mode_to_wav_path,
    }
    save_metadata_json(OUT_META_JSON, meta_payload)
    print(f"Saved meta JSON             : {OUT_META_JSON}")

    print("\nDone.")
    print("Generated modes:")
    for mode_name in mode_to_csv_path:
        print(f"  - {mode_name}")
        print(f"      CSV: {mode_to_csv_path[mode_name]}")
        print(f"      WAV: {mode_to_wav_path[mode_name]}")


def run_reconstruct_compact_csv() -> None:
    print("=" * 80)
    print("RECONSTRUCTION FROM COMPACT CSV")
    print("=" * 80)

    if GENERATION_MODE == "all":
        compact_csv_path = OUTPUT_DIR / f"{RUN_PREFIX}_formula_based.csv"
    else:
        compact_csv_path = OUTPUT_DIR / f"{RUN_PREFIX}_{GENERATION_MODE}.csv"

    print(f"Compact CSV: {compact_csv_path}")

    df = pd.read_csv(compact_csv_path)
    df = enforce_dense_compact_layout(df)

    print("Reconstructing audio from compact CSV only...")
    y, sr = reconstruct_from_compact_csv(
        df=df,
        out_wav=RECON_COMPACT_WAV,
        noise_scale=DEFAULT_NOISE_SCALE,
    )

    print("\nDone.")
    print(f"Output WAV: {RECON_COMPACT_WAV}")
    print(f"Samples   : {len(y)}")
    print(f"SR        : {sr}")


def run_reconstruct_pitch_loudness_csv() -> None:
    print("=" * 80)
    print("RECONSTRUCTION FROM PITCH + LOUDNESS CSV")
    print("=" * 80)
    print(f"Pitch+loudness CSV: {OUT_PITCH_LOUDNESS_CSV}")
    print(f"Checkpoint         : {CKPT_PATH}")

    df = pd.read_csv(OUT_PITCH_LOUDNESS_CSV)

    print("Reconstructing audio from pitch+loudness CSV only...")
    y, sr = reconstruct_from_pitch_loudness_csv(
        df=df,
        ckpt_path=CKPT_PATH,
        out_wav=RECON_PITCH_LOUDNESS_WAV,
        noise_scale=PITCH_LOUDNESS_ONLY_NOISE_SCALE,
    )

    print("\nDone.")
    print(f"Output WAV: {RECON_PITCH_LOUDNESS_WAV}")
    print(f"Samples   : {len(y)}")
    print(f"SR        : {sr}")


# =========================================================
# MAIN
# =========================================================
def main() -> None:
    ensure_dir(OUTPUT_DIR)

    if RUN_MODE == "generate":
        run_generate()
    elif RUN_MODE == "reconstruct_compact_csv":
        run_reconstruct_compact_csv()
    elif RUN_MODE == "reconstruct_pitch_loudness_csv":
        run_reconstruct_pitch_loudness_csv()
    else:
        raise ValueError(f"Unknown RUN_MODE: {RUN_MODE}")


if __name__ == "__main__":
    main()