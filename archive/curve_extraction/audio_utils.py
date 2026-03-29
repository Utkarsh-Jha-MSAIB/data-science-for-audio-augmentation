import numpy as np
from scipy.io.wavfile import write


def moving_average(x, win):
    if win <= 1:
        return x.copy()
    kernel = np.ones(win, dtype=np.float32) / win
    return np.convolve(x, kernel, mode="same")


def normalize_audio(x, peak=0.95):
    x = np.asarray(x, dtype=np.float32)
    m = np.max(np.abs(x))
    if m > 0:
        x = x / m * peak
    return x


def prepare_curve_signal(x, y, sr=16000, duration=5.0, smooth_window=101):
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    y = np.asarray(y, dtype=np.float32).reshape(-1)

    if len(x) != len(y):
        raise ValueError("x and y must have same length")
    if len(x) < 2:
        raise ValueError("Need at least 2 points")

    idx = np.argsort(x)
    x = x[idx]
    y = y[idx]

    xmin, xmax = x.min(), x.max()
    if xmax <= xmin:
        raise ValueError("x must span a non-zero range")

    n_samples = int(sr * duration)
    t_audio = np.linspace(0, duration, n_samples, endpoint=False)
    t_src = (x - xmin) / (xmax - xmin) * duration

    sig = np.interp(t_audio, t_src, y)

    if smooth_window and smooth_window > 1:
        sig = moving_average(sig, smooth_window)

    ymin, ymax = sig.min(), sig.max()
    if ymax > ymin:
        sig = (sig - ymin) / (ymax - ymin)
    else:
        sig = np.zeros_like(sig)

    return sig.astype(np.float32)


def quantize_to_minor_pentatonic(freq_hz):
    """
    Quantize frequencies to A minor pentatonic-ish note set across octaves.
    """
    freq_hz = np.asarray(freq_hz, dtype=np.float32)
    midi = 69 + 12 * np.log2(np.maximum(freq_hz, 1e-6) / 440.0)

    allowed_pc = np.array([0, 3, 5, 7, 10], dtype=np.float32)  # minor pentatonic
    midi_q = np.zeros_like(midi)

    for i, m in enumerate(midi):
        octave = np.floor(m / 12.0)
        pcs = allowed_pc + 12.0 * octave
        # also check neighboring octaves
        candidates = np.concatenate([pcs - 12, pcs, pcs + 12])
        midi_q[i] = candidates[np.argmin(np.abs(candidates - m))]

    return 440.0 * (2.0 ** ((midi_q - 69.0) / 12.0))


def sonify_curve_as_guitar(
    x,
    y,
    out_wav,
    out_env_npy=None,
    out_pitch_npy=None,
    sr=16000,
    duration=5.0,
    smooth_window=101,
    fmin=82.41,    # low E guitar
    fmax=659.25,   # E5-ish upper melodic area
    quantize_scale=True,
):
    """
    Convert curve into a guitar-like additive synthesis sound.
    """
    env = prepare_curve_signal(x, y, sr=sr, duration=duration, smooth_window=smooth_window)

    # Pitch from same curve, but smoother than amplitude
    pitch_driver = moving_average(env, 401)
    pitch_hz = fmin + pitch_driver * (fmax - fmin)

    if quantize_scale:
        pitch_hz = quantize_to_minor_pentatonic(pitch_hz)

    n = len(env)
    t = np.arange(n, dtype=np.float32) / sr

    # phase accumulation for variable frequency
    phase = 2 * np.pi * np.cumsum(pitch_hz) / sr

    # guitar-like harmonic recipe
    sig = (
        1.00 * np.sin(phase) +
        0.45 * np.sin(2 * phase) +
        0.22 * np.sin(3 * phase) +
        0.10 * np.sin(4 * phase) +
        0.05 * np.sin(5 * phase)
    )

    # light pluck/transient emphasis from positive envelope changes
    diff = np.diff(env, prepend=env[0])
    attack = np.clip(diff, 0, None)
    attack = moving_average(attack, 11)
    if attack.max() > 0:
        attack = attack / (attack.max() + 1e-8)

    shaped_env = 0.8 * env + 0.2 * attack

    # tiny noise for pick texture
    noise = np.random.randn(n).astype(np.float32) * 0.01

    audio = sig * shaped_env + noise * attack
    audio = normalize_audio(audio, peak=0.95)

    write(out_wav, sr, (audio * 32767).astype(np.int16))

    if out_env_npy is not None:
        np.save(out_env_npy, env.astype(np.float32))
    if out_pitch_npy is not None:
        np.save(out_pitch_npy, pitch_hz.astype(np.float32))

    return audio, env, pitch_hz, sr