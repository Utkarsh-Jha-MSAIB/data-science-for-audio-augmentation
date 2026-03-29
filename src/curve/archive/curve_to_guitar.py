import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import soundfile as sf
import matplotlib.pyplot as plt

# -------------------------------------------------
# PATH SETUP
# -------------------------------------------------
HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[2]          # experiments -> src -> core
PROJECT_ROOT = REPO_ROOT.parent      # Music_AI_1.1

CURVE_SIMPLE_ROOT = Path(r"C:\curve_simple")
OUTPUT_DIR = CURVE_SIMPLE_ROOT / "output"
SMOOTH_CSV = OUTPUT_DIR / "curve_mapping_smooth.csv"

CKPT_PATH = PROJECT_ROOT / "core" / "checkpoints" / "Guitar.ckpt"

# -------------------------------------------------
# AUDIO / MODEL CONFIG
# -------------------------------------------------
SR = 16000
FRAME_RATE = 250
HOP_SAMPLES = 64
DURATION_SEC = 20.0

FMIN = 82.41     # E2
FMAX = 659.25    # E5
NOISE_SCALE = 0.03

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------------------------
# QUANTIZATION CONFIG
# -------------------------------------------------
QUANTIZER_MODES = [
    "minor_pentatonic",
    "major_pentatonic",
    # "natural_minor",
    # "major",
    # "chromatic",
    # "blues_minor",
]

# C=0, C#=1, D=2, ... A=9, B=11
QUANTIZER_ROOT_PC = 0

USE_STICKY_QUANTIZER = True
STICK_THRESHOLD = 0.75
PREV_PENALTY = 0.35

# smoothing before quantization
LOUD_SMOOTH_WIN = 11
PITCH_SMOOTH_WIN = 121

# -------------------------------------------------
# IMPORT REPO MODULES
# -------------------------------------------------
sys.path.append(str(REPO_ROOT))

from src.models.train_instrument import AudioSynthTrainer
from src.models.signal_processing import harmonic_synthesis, noise_synthesis


# -------------------------------------------------
# HELPERS
# -------------------------------------------------
def moving_average(x, win):
    if win <= 1:
        return x.copy()
    kernel = np.ones(win, dtype=np.float32) / win
    return np.convolve(x, kernel, mode="same")


def normalize_01(x):
    x = np.asarray(x, dtype=np.float32)
    xmin, xmax = x.min(), x.max()
    if xmax > xmin:
        return (x - xmin) / (xmax - xmin)
    return np.zeros_like(x, dtype=np.float32)


def hz_to_midi(freq_hz):
    freq_hz = np.asarray(freq_hz, dtype=np.float32)
    return 69.0 + 12.0 * np.log2(np.maximum(freq_hz, 1e-6) / 440.0)


def midi_to_hz(midi):
    midi = np.asarray(midi, dtype=np.float32)
    return 440.0 * (2.0 ** ((midi - 69.0) / 12.0))


def get_scale_pitch_classes(mode: str):
    scale_map = {
        "minor_pentatonic": np.array([0, 3, 5, 7, 10], dtype=np.float32),
        "major_pentatonic": np.array([0, 2, 4, 7, 9], dtype=np.float32),
        "natural_minor":    np.array([0, 2, 3, 5, 7, 8, 10], dtype=np.float32),
        "major":            np.array([0, 2, 4, 5, 7, 9, 11], dtype=np.float32),
        "chromatic":        np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], dtype=np.float32),
        "blues_minor":      np.array([0, 3, 5, 6, 7, 10], dtype=np.float32),
    }

    if mode not in scale_map:
        valid = ", ".join(scale_map.keys())
        raise ValueError(f"Unknown quantizer mode '{mode}'. Valid options: {valid}")

    return scale_map[mode]


def quantize_to_scale(
    freq_hz,
    mode="minor_pentatonic",
    root_pc=0,
    sticky=False,
    stick_threshold=0.75,
    prev_penalty=0.35,
):
    freq_hz = np.asarray(freq_hz, dtype=np.float32)
    midi = hz_to_midi(freq_hz)

    base_pc = get_scale_pitch_classes(mode)
    allowed_pc = (base_pc + float(root_pc)) % 12.0

    midi_q = np.zeros_like(midi, dtype=np.float32)

    prev_q = None
    for i, m in enumerate(midi):
        octave = np.floor(m / 12.0)
        pcs = allowed_pc + 12.0 * octave
        candidates = np.concatenate([pcs - 12.0, pcs, pcs + 12.0])

        if (not sticky) or (prev_q is None):
            chosen = candidates[np.argmin(np.abs(candidates - m))]
        else:
            if np.abs(m - prev_q) < stick_threshold:
                chosen = prev_q
            else:
                costs = np.abs(candidates - m) + prev_penalty * np.abs(candidates - prev_q)
                chosen = candidates[np.argmin(costs)]

        midi_q[i] = chosen
        prev_q = chosen

    return midi_to_hz(midi_q).astype(np.float32)


def load_audio_model(ckpt_path: Path):
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing model checkpoint: {ckpt_path}")

    print(f"Loading checkpoint from: {ckpt_path}")
    model = AudioSynthTrainer.load_from_checkpoint(
        str(ckpt_path),
        map_location=torch.device("cpu")
    )
    model.eval()
    return model


def save_control_plot(pitch_hz, loudness, out_path, title_suffix=""):
    fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    ax[0].plot(pitch_hz, linewidth=1.5)
    ax[0].set_title(f"Pitch control (Hz){title_suffix}")
    ax[0].set_ylabel("Hz")
    ax[0].grid(alpha=0.3)

    ax[1].plot(loudness, linewidth=1.5)
    ax[1].set_title("Loudness control")
    ax[1].set_ylabel("Normalized")
    ax[1].set_xlabel("Frame")
    ax[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_pitch_comparison_plot(results_dict, out_path):
    fig, ax = plt.subplots(figsize=(14, 6))

    for mode, result in results_dict.items():
        ax.plot(result["pitch_hz"], linewidth=1.2, label=mode)

    ax.set_title("Pitch comparison across quantization modes")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Hz")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def build_base_controls(
    smooth_df: pd.DataFrame,
    duration_sec: float,
    frame_rate: int = 250,
):
    required_cols = {"x_px", "displacement_px"}
    missing = required_cols - set(smooth_df.columns)
    if missing:
        raise ValueError(f"Missing required columns in smooth_df: {missing}")

    x = smooth_df["x_px"].to_numpy(dtype=np.float32)
    y = smooth_df["displacement_px"].to_numpy(dtype=np.float32)

    if len(x) < 2:
        raise ValueError("Curve file does not contain enough points.")

    idx = np.argsort(x)
    x = x[idx]
    y = y[idx]

    y_norm = normalize_01(y)

    y_loud = moving_average(y_norm, LOUD_SMOOTH_WIN)
    y_pitch = moving_average(y_norm, PITCH_SMOOTH_WIN)

    n_frames = int(duration_sec * frame_rate)

    t_src = np.linspace(0, duration_sec, len(y_norm), endpoint=False)
    t_dst = np.linspace(0, duration_sec, n_frames, endpoint=False)

    loudness = np.interp(t_dst, t_src, y_loud).astype(np.float32)
    pitch_driver = np.interp(t_dst, t_src, y_pitch).astype(np.float32)

    return pitch_driver, loudness


def curve_to_model_controls_for_mode(
    smooth_df: pd.DataFrame,
    quantizer_mode: str,
    duration_sec: float,
    frame_rate: int = 250,
    fmin: float = 82.41,
    fmax: float = 659.25,
    root_pc: int = 0,
    sticky: bool = True,
    stick_threshold: float = 0.75,
    prev_penalty: float = 0.35,
):
    pitch_driver, loudness = build_base_controls(
        smooth_df,
        duration_sec=duration_sec,
        frame_rate=frame_rate,
    )

    raw_pitch_hz = fmin + pitch_driver * (fmax - fmin)

    pitch_hz = quantize_to_scale(
        raw_pitch_hz,
        mode=quantizer_mode,
        root_pc=root_pc,
        sticky=sticky,
        stick_threshold=stick_threshold,
        prev_penalty=prev_penalty,
    ).astype(np.float32)

    loudness = np.clip(loudness, 0.0, 1.0)
    loudness = 0.10 + 0.70 * loudness

    return pitch_hz, loudness


def synthesize_from_controls(
    model,
    pitch_hz,
    loudness,
    sr=16000,
    hop_samples=64,
    noise_scale=0.03,
):
    p_t = torch.from_numpy(pitch_hz).float().reshape(1, -1, 1).to(DEVICE)
    l_t = torch.from_numpy(loudness).float().reshape(1, -1, 1).to(DEVICE)

    print("Pitch tensor shape:", tuple(p_t.shape))
    print("Loudness tensor shape:", tuple(l_t.shape))

    with torch.no_grad():
        controls = model.model(p_t, l_t)

        print("Model control keys:", list(controls.keys()))
        for k, v in controls.items():
            if torch.is_tensor(v):
                print(f"  {k}: {tuple(v.shape)}")

        amplitudes = controls.get("amplitudes", None)
        harmonics = controls.get("harmonics", None)
        noise = controls.get("noise", None)

        if amplitudes is None:
            raise KeyError("Model output missing 'amplitudes'")
        if harmonics is None:
            raise KeyError("Model output missing 'harmonics'")

        if noise is not None:
            noise = noise * noise_scale

        n_samples = p_t.shape[1] * hop_samples

        harm = harmonic_synthesis(
            p_t,
            amplitudes,
            harmonics,
            n_samples=n_samples,
            sample_rate=sr
        )

        if noise is not None:
            nz = noise_synthesis(noise, n_samples=n_samples)
            audio = harm + nz
        else:
            audio = harm

    wav = audio.squeeze().detach().cpu().numpy().astype(np.float32)

    peak = np.max(np.abs(wav))
    if peak > 0:
        wav = wav / peak * 0.95

    return wav


# -------------------------------------------------
# MAIN
# -------------------------------------------------
def main():
    print(f"Repo root: {REPO_ROOT}")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Curve CSV: {SMOOTH_CSV}")
    print(f"Checkpoint: {CKPT_PATH}")
    print(f"Device: {DEVICE}")

    if not SMOOTH_CSV.exists():
        raise FileNotFoundError(f"Missing smooth curve CSV: {SMOOTH_CSV}")

    if not CKPT_PATH.exists():
        raise FileNotFoundError(f"Missing Guitar checkpoint: {CKPT_PATH}")

    smooth_df = pd.read_csv(SMOOTH_CSV)
    print("Loaded curve columns:", smooth_df.columns.tolist())
    print("Number of curve rows:", len(smooth_df))

    model = load_audio_model(CKPT_PATH).to(DEVICE)

    results = {}

    for mode in QUANTIZER_MODES:
        print("\n" + "=" * 70)
        print(f"Processing quantizer mode: {mode}")
        print("=" * 70)

        pitch_hz, loudness = curve_to_model_controls_for_mode(
            smooth_df,
            quantizer_mode=mode,
            duration_sec=DURATION_SEC,
            frame_rate=FRAME_RATE,
            fmin=FMIN,
            fmax=FMAX,
            root_pc=QUANTIZER_ROOT_PC,
            sticky=USE_STICKY_QUANTIZER,
            stick_threshold=STICK_THRESHOLD,
            prev_penalty=PREV_PENALTY,
        )

        mode_prefix = f"guitar_ckpt_curve_{mode}"

        out_wav = OUTPUT_DIR / f"{mode_prefix}.wav"
        out_pitch = OUTPUT_DIR / f"{mode_prefix}_pitch.npy"
        out_loud = OUTPUT_DIR / f"{mode_prefix}_loudness.npy"
        out_plot = OUTPUT_DIR / f"{mode_prefix}_controls.png"

        np.save(out_pitch, pitch_hz)
        np.save(out_loud, loudness)
        save_control_plot(
            pitch_hz,
            loudness,
            out_plot,
            title_suffix=f" - {mode}"
        )

        print(f"Saved pitch NPY: {out_pitch}")
        print(f"Saved loudness NPY: {out_loud}")
        print(f"Saved control plot: {out_plot}")

        print("Synthesizing audio...")
        wav = synthesize_from_controls(
            model,
            pitch_hz,
            loudness,
            sr=SR,
            hop_samples=HOP_SAMPLES,
            noise_scale=NOISE_SCALE,
        )
        sf.write(out_wav, wav, SR)

        print(f"Saved WAV: {out_wav}")

        results[mode] = {
            "pitch_hz": pitch_hz,
            "loudness": loudness,
            "wav_path": out_wav,
            "pitch_path": out_pitch,
            "loud_path": out_loud,
            "plot_path": out_plot,
        }

    comparison_plot = OUTPUT_DIR / "guitar_ckpt_all_quantizers_pitch_comparison.png"
    save_pitch_comparison_plot(results, comparison_plot)
    print(f"\nSaved overall pitch comparison plot: {comparison_plot}")

    print("\nDone. Generated outputs for all quantizer modes.")


if __name__ == "__main__":
    main()