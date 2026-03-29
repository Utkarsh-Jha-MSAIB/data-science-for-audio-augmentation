import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch

# =========================================================
# PATH SETUP
# =========================================================
HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[2]          # experiments -> src -> core
PROJECT_ROOT = REPO_ROOT.parent      # Music_AI_1.1

# =========================================================
# IMPORT REPO MODULES
# =========================================================
sys.path.append(str(REPO_ROOT))

from src.models.train_instrument import AudioSynthTrainer
from src.models.signal_processing import harmonic_synthesis, noise_synthesis

# =========================================================
# USER CONFIG
# =========================================================
RUN_MODE = "reconstruct_compact_csv"
# choices:
#   "extract"
#   "reconstruct_compact_csv"
#   "reconstruct_full_csv"
#   "reconstruct_pitch_loudness_csv"

INPUT_WAV = Path(r"C:\Music_AI_1.1\outputs\input.wav")
OUTPUT_DIR = Path(r"C:\Music_AI_1.1\outputs\csv_reconstruction")
CKPT_PATH = PROJECT_ROOT / "core" / "checkpoints" / "Guitar.ckpt"

COMPACT_CSV_PATH = OUTPUT_DIR / "audio_compact_reconstruction_bundle.csv"
FULL_CSV_PATH = OUTPUT_DIR / "audio_full_reconstruction_bundle.csv"
PITCH_LOUDNESS_CSV_PATH = OUTPUT_DIR / "audio_pitch_loudness_bundle.csv"

META_JSON_PATH = OUTPUT_DIR / "audio_reconstruction_meta.json"
SUMMARY_PNG_PATH = OUTPUT_DIR / "audio_reconstruction_summary.png"

RECON_COMPACT_WAV = OUTPUT_DIR / "reconstructed_from_compact_csv.wav"
RECON_FULL_WAV = OUTPUT_DIR / "reconstructed_from_full_csv.wav"
RECON_PITCH_LOUDNESS_WAV = OUTPUT_DIR / "reconstructed_from_pitch_loudness_csv.wav"

ROUND_FLOATS_FOR_CSV = False
CSV_FLOAT_DECIMALS = 6

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
UNVOICED_FILL_MODE = "nearest"   # nearest | median
PITCH_LOUDNESS_ONLY_NOISE_SCALE = 0.03

# Compact-control estimation from original audio
NUM_HARMONICS = 100
NUM_NOISE_BANDS = 64
HARMONIC_SAMPLE_BINS = 1
HARMONIC_ENERGY_SMOOTH_WIN = 5
NOISE_ENERGY_SMOOTH_WIN = 7

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
    float_cols = df.select_dtypes(include=["float32", "float64"]).columns
    df = df.copy()
    df[float_cols] = df[float_cols].round(CSV_FLOAT_DECIMALS)
    return df


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


# =========================================================
# ANALYSIS EXTRACTION
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


# =========================================================
# CONTROL GENERATION / ESTIMATION
# =========================================================
def predict_synth_controls_from_pitch_loudness(
    model: AudioSynthTrainer,
    pitch_hz_for_model: np.ndarray,
    loudness_for_model: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Given only pitch + loudness, ask the checkpoint to regenerate:
    amplitudes, harmonics, noise.
    """
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
        "amplitudes": amplitudes.squeeze(0).detach().cpu().numpy().astype(np.float32),   # [T,1]
        "harmonics": harmonics.squeeze(0).detach().cpu().numpy().astype(np.float32),      # [T,H]
        "noise": None if noise is None else noise.squeeze(0).detach().cpu().numpy().astype(np.float32),  # [T,C]
    }


def estimate_compact_controls_from_audio(
    s_complex: np.ndarray,
    attrs: Dict[str, np.ndarray],
    meta: AnalysisMeta,
    num_harmonics: int = NUM_HARMONICS,
    num_noise_bands: int = NUM_NOISE_BANDS,
) -> Dict[str, np.ndarray]:
    """
    Estimate compact controls from the original audio.

    This makes compact CSV different from pitch+loudness CSV:
    - compact stores audio-derived amplitudes/harmonics/noise
    - pitch+loudness CSV stores only pitch+loudness, and reconstructs by re-predicting
    """
    pitch_hz = np.asarray(attrs["f0_hz"], dtype=np.float32)
    loudness = np.asarray(attrs["rms_norm"], dtype=np.float32)
    pitch_hz_for_model = fill_unvoiced_pitch(pitch_hz, mode=UNVOICED_FILL_MODE)

    mag = np.abs(s_complex).T.astype(np.float32)   # [T, F]
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
# DATAFRAME BUILDERS
# =========================================================
def build_pitch_loudness_dataframe(
    attrs: Dict[str, np.ndarray],
    meta: AnalysisMeta,
) -> pd.DataFrame:
    n_frames = len(attrs["frame_idx"])
    pitch_hz_for_model = fill_unvoiced_pitch(
        np.asarray(attrs["f0_hz"], dtype=np.float32),
        mode=UNVOICED_FILL_MODE,
    )
    loudness_for_model = np.asarray(attrs["rms_norm"], dtype=np.float32)

    return pd.DataFrame({
        "frame_idx": attrs["frame_idx"].astype(np.int32),
        "sr": np.full(n_frames, meta.sr, dtype=np.int32),
        "hop_length": np.full(n_frames, meta.hop_length, dtype=np.int32),
        "pitch_hz_for_model": pitch_hz_for_model.astype(np.float32),
        "loudness_for_model": loudness_for_model.astype(np.float32),
    })


def build_compact_dataframe(
    attrs: Dict[str, np.ndarray],
    compact_controls: Dict[str, np.ndarray],
    meta: AnalysisMeta,
) -> pd.DataFrame:
    n_frames = len(attrs["frame_idx"])

    base_df = pd.DataFrame({
        "frame_idx": attrs["frame_idx"].astype(np.int32),
        "sr": np.full(n_frames, meta.sr, dtype=np.int32),
        "hop_length": np.full(n_frames, meta.hop_length, dtype=np.int32),
        "pitch_hz_for_model": compact_controls["pitch_hz_for_model"].astype(np.float32),
        "loudness_for_model": compact_controls["loudness_for_model"].astype(np.float32),
        "amplitudes": compact_controls["amplitudes"].reshape(-1).astype(np.float32),
    })

    pieces = [base_df]

    harmonics = compact_controls["harmonics"].astype(np.float32)
    harmonic_cols = [f"harmonic_{h:03d}" for h in range(harmonics.shape[1])]
    pieces.append(pd.DataFrame(harmonics, columns=harmonic_cols))

    if compact_controls["noise"] is not None:
        noise = compact_controls["noise"].astype(np.float32)
        noise_cols = [f"noise_{c:03d}" for c in range(noise.shape[1])]
        pieces.append(pd.DataFrame(noise, columns=noise_cols))

    return pd.concat(pieces, axis=1, copy=False)


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
    harmonic_cols = [f"harmonic_{h:03d}" for h in range(harmonics.shape[1])]
    pieces.append(pd.DataFrame(harmonics, columns=harmonic_cols))

    if compact_controls["noise"] is not None:
        noise = compact_controls["noise"].astype(np.float32)
        noise_cols = [f"noise_{c:03d}" for c in range(noise.shape[1])]
        pieces.append(pd.DataFrame(noise, columns=noise_cols))

    stft_real = np.real(s_complex).T.astype(np.float32)
    stft_imag = np.imag(s_complex).T.astype(np.float32)

    real_cols = [f"stft_real_bin_{b:04d}" for b in range(n_freq_bins)]
    imag_cols = [f"stft_imag_bin_{b:04d}" for b in range(n_freq_bins)]

    pieces.append(pd.DataFrame(stft_real, columns=real_cols))
    pieces.append(pd.DataFrame(stft_imag, columns=imag_cols))

    return pd.concat(pieces, axis=1, copy=False)


# =========================================================
# RECONSTRUCTION HELPERS
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

        if noise is not None:
            n_t = torch.from_numpy(noise).float().reshape(1, noise.shape[0], noise.shape[1]).to(DEVICE)
            nz = noise_synthesis(n_t * noise_scale, n_samples=n_samples)
            audio = harm #+ nz
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
    df = df.sort_values("frame_idx").reset_index(drop=True)

    sr = int(df["sr"].iloc[0])
    hop_length = int(df["hop_length"].iloc[0])

    pitch_hz_for_model = df["pitch_hz_for_model"].to_numpy(dtype=np.float32)
    loudness_for_model = df["loudness_for_model"].to_numpy(dtype=np.float32)
    amplitudes = df["amplitudes"].to_numpy(dtype=np.float32).reshape(-1, 1)

    harmonic_cols = [c for c in df.columns if c.startswith("harmonic_")]
    noise_cols = [c for c in df.columns if c.startswith("noise_")]

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


def reconstruct_from_full_csv(df: pd.DataFrame, out_wav: Path) -> Tuple[np.ndarray, int]:
    df = df.sort_values("frame_idx").reset_index(drop=True)

    n_freq_bins = int(df["n_freq_bins"].iloc[0])
    hop_length = int(df["hop_length"].iloc[0])
    win_length = int(df["win_length"].iloc[0])
    num_samples = int(df["num_samples"].iloc[0])
    sr = int(df["sr"].iloc[0])

    real_cols = [f"stft_real_bin_{b:04d}" for b in range(n_freq_bins)]
    imag_cols = [f"stft_imag_bin_{b:04d}" for b in range(n_freq_bins)]

    missing_real = [c for c in real_cols if c not in df.columns]
    missing_imag = [c for c in imag_cols if c not in df.columns]
    if missing_real or missing_imag:
        raise ValueError("Full CSV is missing STFT real/imag columns.")

    stft_real = df[real_cols].to_numpy(dtype=np.float32).T
    stft_imag = df[imag_cols].to_numpy(dtype=np.float32).T
    s_complex = stft_real + 1j * stft_imag

    y = librosa.istft(
        s_complex,
        hop_length=hop_length,
        win_length=win_length,
        window=WINDOW,
        center=True,
        length=num_samples,
    ).astype(np.float32, copy=False)

    peak = float(np.max(np.abs(y)))
    if peak > 1.0:
        y = y / peak * 0.999

    sf.write(out_wav, y, sr)
    return y, sr


# =========================================================
# REPORTING
# =========================================================
def save_metadata_json(meta_path: Path, meta: AnalysisMeta) -> None:
    meta_dict = {
        "sr": meta.sr,
        "duration_sec": meta.duration_sec,
        "num_samples": meta.num_samples,
        "n_fft": meta.n_fft,
        "hop_length": meta.hop_length,
        "win_length": meta.win_length,
        "window": meta.window,
        "n_frames": meta.n_frames,
        "n_freq_bins": meta.n_freq_bins,
        "frame_rate": meta.frame_rate,
        "source_wav": meta.source_wav,
        "ckpt_path": meta.ckpt_path,
    }
    meta_path.write_text(json.dumps(meta_dict, indent=2))


def plot_summary(full_df: pd.DataFrame, plot_path: Path) -> None:
    import matplotlib.pyplot as plt

    t = full_df["time_sec"].to_numpy(dtype=np.float32)

    harmonic_cols = [c for c in full_df.columns if c.startswith("harmonic_")]
    noise_cols = [c for c in full_df.columns if c.startswith("noise_")]

    harmonic_energy = full_df[harmonic_cols].sum(axis=1).to_numpy(dtype=np.float32) if harmonic_cols else np.zeros_like(t)
    noise_mean = full_df[noise_cols].mean(axis=1).to_numpy(dtype=np.float32) if noise_cols else np.zeros_like(t)

    fig, ax = plt.subplots(5, 1, figsize=(14, 12), sharex=True)

    ax[0].plot(t, full_df["rms_norm"].to_numpy(dtype=np.float32), linewidth=1.0)
    ax[0].set_title("RMS Normalized")
    ax[0].grid(alpha=0.3)

    ax[1].plot(t, full_df["f0_hz"].to_numpy(dtype=np.float32), linewidth=1.0)
    ax[1].set_title("F0 (Hz)")
    ax[1].grid(alpha=0.3)

    ax[2].plot(t, full_df["amplitudes"].to_numpy(dtype=np.float32), linewidth=1.0)
    ax[2].set_title("Compact Amplitude")
    ax[2].grid(alpha=0.3)

    ax[3].plot(t, harmonic_energy, linewidth=1.0)
    ax[3].set_title("Compact Harmonic Energy")
    ax[3].grid(alpha=0.3)

    ax[4].plot(t, noise_mean, linewidth=1.0, label="Compact noise mean")
    ax[4].legend()
    ax[4].set_title("Compact Noise")
    ax[4].set_xlabel("Time (sec)")
    ax[4].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)


# =========================================================
# PIPELINE MODES
# =========================================================
def run_extract() -> None:
    ensure_dir(OUTPUT_DIR)

    print("=" * 80)
    print("AUDIO ANALYSIS + CSV BUNDLE EXTRACTION")
    print("=" * 80)
    print(f"Input WAV       : {INPUT_WAV}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Checkpoint      : {CKPT_PATH}")
    print(f"Device          : {DEVICE}")

    print("\n[1/6] Loading audio...")
    y = load_audio_mono(INPUT_WAV, sr=SR)

    print("[2/6] Computing analysis features...")
    s_complex, attrs, meta = compute_framewise_attributes(
        y=y,
        input_wav=INPUT_WAV,
        ckpt_path=CKPT_PATH,
        sr=SR,
    )

    print("[3/6] Estimating compact controls directly from audio...")
    compact_controls = estimate_compact_controls_from_audio(
        s_complex=s_complex,
        attrs=attrs,
        meta=meta,
        num_harmonics=NUM_HARMONICS,
        num_noise_bands=NUM_NOISE_BANDS,
    )

    print("[4/6] Building pitch+loudness-only CSV...")
    pitch_loudness_df = build_pitch_loudness_dataframe(attrs, meta)

    print("[5/6] Building compact + full CSV bundles...")
    compact_df = build_compact_dataframe(attrs, compact_controls, meta)
    full_df = build_full_dataframe(s_complex, attrs, compact_controls, meta)

    if ROUND_FLOATS_FOR_CSV:
        pitch_loudness_df = maybe_round_df(pitch_loudness_df)
        compact_df = maybe_round_df(compact_df)
        full_df = maybe_round_df(full_df)

    print("[6/6] Saving outputs...")
    pitch_loudness_df.to_csv(PITCH_LOUDNESS_CSV_PATH, index=False)
    compact_df.to_csv(COMPACT_CSV_PATH, index=False)
    full_df.to_csv(FULL_CSV_PATH, index=False)
    save_metadata_json(META_JSON_PATH, meta)
    plot_summary(full_df, SUMMARY_PNG_PATH)

    print("\nDone.")
    print(f"Pitch+loudness CSV rows/cols: {pitch_loudness_df.shape}")
    print(f"Compact CSV rows/cols       : {compact_df.shape}")
    print(f"Full CSV rows/cols          : {full_df.shape}")
    print(f"Pitch+loudness CSV          : {PITCH_LOUDNESS_CSV_PATH}")
    print(f"Compact CSV                 : {COMPACT_CSV_PATH}")
    print(f"Full CSV                    : {FULL_CSV_PATH}")
    print(f"Meta JSON                   : {META_JSON_PATH}")
    print(f"Summary plot                : {SUMMARY_PNG_PATH}")


def run_reconstruct_compact_csv() -> None:
    print("=" * 80)
    print("RECONSTRUCTION FROM COMPACT CSV")
    print("=" * 80)
    print(f"Compact CSV: {COMPACT_CSV_PATH}")

    df = pd.read_csv(COMPACT_CSV_PATH)

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


def run_reconstruct_full_csv() -> None:
    print("=" * 80)
    print("RECONSTRUCTION FROM FULL CSV")
    print("=" * 80)
    print(f"Full CSV: {FULL_CSV_PATH}")

    df = pd.read_csv(FULL_CSV_PATH)

    print("Reconstructing exact audio from full CSV only...")
    y, sr = reconstruct_from_full_csv(
        df=df,
        out_wav=RECON_FULL_WAV,
    )

    print("\nDone.")
    print(f"Output WAV: {RECON_FULL_WAV}")
    print(f"Samples   : {len(y)}")
    print(f"SR        : {sr}")


def run_reconstruct_pitch_loudness_csv() -> None:
    print("=" * 80)
    print("RECONSTRUCTION FROM PITCH + LOUDNESS CSV")
    print("=" * 80)
    print(f"Pitch+loudness CSV: {PITCH_LOUDNESS_CSV_PATH}")
    print(f"Checkpoint         : {CKPT_PATH}")

    df = pd.read_csv(PITCH_LOUDNESS_CSV_PATH)

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

    if RUN_MODE == "extract":
        run_extract()
    elif RUN_MODE == "reconstruct_compact_csv":
        run_reconstruct_compact_csv()
    elif RUN_MODE == "reconstruct_full_csv":
        run_reconstruct_full_csv()
    elif RUN_MODE == "reconstruct_pitch_loudness_csv":
        run_reconstruct_pitch_loudness_csv()
    else:
        raise ValueError(f"Unknown RUN_MODE: {RUN_MODE}")


if __name__ == "__main__":
    main()