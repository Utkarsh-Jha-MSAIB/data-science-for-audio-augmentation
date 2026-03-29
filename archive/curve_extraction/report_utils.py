import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


FRAME_LENGTH = 2048
HOP_LENGTH = 64


def calculate_guitar_scores(f0_plot, env_plot):
    energy_score = int(np.clip(np.mean(env_plot) * 100, 0, 100))
    dynamics_score = int(np.clip((np.max(env_plot) - np.min(env_plot)) * 100, 0, 100))

    active = f0_plot[f0_plot > 0]
    if len(active) > 0:
        complexity_score = int(np.clip(np.std(active) / 4.0, 0, 100))
    else:
        complexity_score = 0

    return {
        "Energy": energy_score,
        "Dynamics": dynamics_score,
        "Complexity": complexity_score,
    }


def save_guitar_report_card(file_path, env_npy_path=None, pitch_npy_path=None):
    path_obj = Path(file_path)

    if not path_obj.exists():
        print(f"Report Card Error: File not found at {file_path}")
        return

    print(f"Generating Guitar Dashboard for: {path_obj.name}...")

    y, sr = librosa.load(str(path_obj), sr=16000)
    duration = librosa.get_duration(y=y, sr=sr)

    rms = librosa.feature.rms(
        y=y,
        frame_length=FRAME_LENGTH,
        hop_length=HOP_LENGTH
    )[0]
    rms_norm = rms / (np.max(rms) + 1e-7)
    times_rms = librosa.times_like(rms_norm, sr=sr, hop_length=HOP_LENGTH)

    # use saved pitch if available; otherwise estimate
    if pitch_npy_path is not None and Path(pitch_npy_path).exists():
        f0 = np.load(pitch_npy_path)
        times_f0 = np.linspace(0, duration, len(f0), endpoint=False)
    else:
        f0, _, _ = librosa.pyin(
            y,
            fmin=librosa.note_to_hz("E2"),
            fmax=librosa.note_to_hz("E5"),
            sr=sr,
            frame_length=FRAME_LENGTH,
            hop_length=HOP_LENGTH
        )
        f0 = np.nan_to_num(f0)
        times_f0 = librosa.times_like(f0, sr=sr, hop_length=HOP_LENGTH)

    if env_npy_path is not None and Path(env_npy_path).exists():
        env = np.load(env_npy_path)
        times_env = np.linspace(0, duration, len(env), endpoint=False)
    else:
        env = rms_norm
        times_env = times_rms

    scores = calculate_guitar_scores(f0, env)

    plt.figure(figsize=(10, 12))
    plt.suptitle(f"Guitar Audio DNA: {path_obj.name}", fontsize=16, fontweight="bold")

    plt.subplot(4, 1, 1)
    librosa.display.waveshow(y, sr=sr, alpha=0.7)
    plt.title("Waveform (Guitar-like Texture)")
    plt.axis("off")

    plt.subplot(4, 1, 2)
    plt.plot(times_f0, f0, linewidth=2)
    plt.title(f"Pitch Curve (Complexity: {scores['Complexity']}/100)")
    plt.ylabel("Hz")
    plt.grid(alpha=0.3)

    plt.subplot(4, 1, 3)
    plt.plot(times_env, env, linewidth=2, label="Target envelope")
    plt.plot(times_rms, rms_norm, alpha=0.6, label="Audio RMS")
    plt.title(
        f"Loudness Envelope "
        f"(Energy: {scores['Energy']}/100, Dynamics: {scores['Dynamics']}/100)"
    )
    plt.ylabel("Normalized Level")
    plt.grid(alpha=0.3)
    plt.legend()

    plt.subplot(4, 1, 4)
    plt.axis("off")
    text_str = (
        f"DURATION:   {duration:.2f} sec\n\n"
        f"ENERGY:     {scores['Energy']} / 100\n"
        f"DYNAMICS:   {scores['Dynamics']} / 100\n"
        f"COMPLEXITY: {scores['Complexity']} / 100"
    )
    plt.text(
        0.5, 0.5, text_str,
        ha="center", va="center", fontsize=14,
        bbox=dict(boxstyle="round,pad=1", fc="#f0f0f0", ec="black", alpha=0.5)
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    output_img = path_obj.with_suffix(".png")
    plt.savefig(output_img, dpi=200)
    plt.close()

    print(f"Guitar Report Card Saved: {output_img}")