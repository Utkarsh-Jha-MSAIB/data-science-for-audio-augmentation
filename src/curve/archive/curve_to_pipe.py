import math
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import matplotlib.pyplot as plt
import librosa

# -------------------------------------------------
# PATH SETUP
# -------------------------------------------------
HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[2]          # experiments -> src -> core
PROJECT_ROOT = REPO_ROOT.parent      # Music_AI_1.1

CURVE_SIMPLE_ROOT = Path(r"C:\curve_simple")
OUTPUT_DIR = CURVE_SIMPLE_ROOT / "output"
SMOOTH_CSV = OUTPUT_DIR / "curve_mapping_smooth.csv"

# -------------------------------------------------
# HARDCODE YOUR REAL PIPE FILES
# -------------------------------------------------
PIPE_AUDIO_PATH = OUTPUT_DIR / "audio_Pipe.npy"
PIPE_LOUD_PATH  = OUTPUT_DIR / "loudness_Pipe.npy"
PIPE_PITCH_PATH = OUTPUT_DIR / "pitch_Pipe.npy"

OUT_WAV = OUTPUT_DIR / "pipe_curve_line_guided.wav"
OUT_NOTE_CSV = OUTPUT_DIR / "pipe_curve_line_guided_notes.csv"
OUT_CTRL_PLOT = OUTPUT_DIR / "pipe_curve_line_guided_controls_debug.png"
OUT_MATCH_PLOT = OUTPUT_DIR / "pipe_curve_line_guided_match.png"

# -------------------------------------------------
# AUDIO CONFIG
# -------------------------------------------------
SR = 16000
FRAME_RATE = 250
HOP_SAMPLES = 64
DURATION_SEC = 30.0

# -------------------------------------------------
# CURVE CONTROL
# -------------------------------------------------
PITCH_SOURCE = "displacement_px"
LOUDNESS_SOURCE = "displacement_px"

PITCH_INVERT = False
LOUDNESS_INVERT = False

# -------------------------------------------------
# PITCH SMOOTHING / MELODY SHAPE
# -------------------------------------------------
PITCH_SMOOTH_WINDOW = 200
MIDI_FLOAT_SMOOTH_WINDOW = 121
MIDI_DEADBAND_SEMITONES = 0.60
EVENT_MIDI_SMOOTH_WINDOW = 5
EVENT_MIDI_SMOOTH_BLEND = 0.70

# Note / melody behavior
NOTE_RATE = 1
MIN_NOTE_DUR_SEC = 0.42
MAX_NOTE_DUR_SEC = 1.10
LOUD_SMOOTH_WINDOW = 10

# Pipe range
PITCH_MIN_MIDI = 62              # D4
PITCH_MAX_MIDI = 79              # G5
SCALE_NAME = "A_minor"
ROOT_MIDI = 57                   # A3 anchor for A minor quantization

# -------------------------------------------------
# HIGH-PITCH CONTROL
# -------------------------------------------------
# Stronger pitch soft ceiling so the upper contour is compressed more clearly
ENABLE_PITCH_SOFT_CEILING = True
PITCH_SOFT_CEILING_THRESHOLD = 0.45
PITCH_SOFT_CEILING_STRENGTH = 0.85
PITCH_SOFT_CEILING_POWER = 1.2

# Hard final cap after quantization so the shrill top region cannot survive
ENABLE_FINAL_MIDI_CAP = True
FINAL_MIDI_CAP = 70

# Optional: slightly reduce loudness when pitch is very high even if co-peak is mild
ENABLE_HIGH_PITCH_LOUDNESS_SOFTEN = True
HIGH_PITCH_LOUDNESS_THRESHOLD = 0.58
HIGH_PITCH_LOUDNESS_STRENGTH = 0.35
HIGH_PITCH_LOUDNESS_POWER = 1.5

# Loudness shaping
LOUD_FLOOR = 0.10
LOUD_RANGE = 0.80
EXPAND_THRESHOLD = 0.35
EXPAND_RATIO = 1.8
ATTACK_BLEND = 0.02
PIPE_VOLUME = 0.92

# -------------------------------------------------
# CO-PEAK SUPPRESSION
# -------------------------------------------------
ENABLE_COPEAK_SUPPRESSION = True

# Start suppressing only when both pitch and loudness are already fairly high
COPEAK_PITCH_THRESHOLD = 0.40
COPEAK_LOUD_THRESHOLD = 0.40

# Must stay <= 1.0
COPEAK_SUPPRESS_STRENGTH = 0.85

# Lower = broader suppression, higher = more focused on extreme overlap
COPEAK_SUPPRESS_POWER = 1.5

# -------------------------------------------------
# SEARCH / RETRIEVAL
# -------------------------------------------------
SEARCH_DS = 8
COARSE_STEP_FRAMES = 64
REFINE_RADIUS_FRAMES = 64
REFINE_STEP_FRAMES = 16
CHUNK_PAD_FRAMES = 2
MIN_VALID_PITCH_FRAMES = 3
MAX_PITCH_STD = 1.5

# Retrieval scoring weights
PITCH_WEIGHT = 4.8
LOUD_WEIGHT = 1.8
STABILITY_WEIGHT = 0.8

# -------------------------------------------------
# LINE-GUIDED SEGMENTATION
# -------------------------------------------------
PITCH_CHANGE_THRESH = 2.4
LOUD_ATTACK_THRESH = 0.14
PLATEAU_HOLD_FRAMES = 30
MAX_MELODIC_JUMP = 3

# -------------------------------------------------
# NOTE RENDERING
# -------------------------------------------------
FADE_MS = 35
USE_WRAP = True
MAX_NOTES = None

# -------------------------------------------------
# PITCH SHIFT CONFIG
# -------------------------------------------------
ENABLE_PITCH_SHIFT = True
MAX_SHIFT_SEMITONES = 2.0
PITCH_SHIFT_CLAMP = True

# -------------------------------------------------
# HELPERS
# -------------------------------------------------
def moving_average(x, win):
    x = np.asarray(x, dtype=np.float32)
    if win <= 1:
        return x.copy()
    if win % 2 == 0:
        win += 1
    kernel = np.ones(win, dtype=np.float32) / win
    return np.convolve(x, kernel, mode="same")


def normalize_01(x):
    x = np.asarray(x, dtype=np.float32)
    xmin, xmax = x.min(), x.max()
    if xmax > xmin:
        return (x - xmin) / (xmax - xmin)
    return np.zeros_like(x, dtype=np.float32)


def flatten_feature_array(arr):
    arr = np.asarray(arr, dtype=np.float32)
    return arr.reshape(-1).astype(np.float32)


def resample_1d(x, target_len):
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if len(x) == target_len:
        return x.copy()
    if len(x) < 2:
        return np.full(target_len, x[0] if len(x) == 1 else 0.0, dtype=np.float32)
    src_t = np.linspace(0, 1, len(x), endpoint=False)
    dst_t = np.linspace(0, 1, target_len, endpoint=False)
    return np.interp(dst_t, src_t, x).astype(np.float32)


def boost_dynamics_np(curve, threshold=0.4, ratio=4.0):
    curve = np.asarray(curve, dtype=np.float32)
    centered = curve - threshold
    expanded = centered * ratio
    result = expanded + threshold
    return np.clip(result, 0.0, 1.0)


def soft_limit_pitch_norm(
    pitch_norm,
    threshold=0.45,
    strength=0.85,
    power=1.2,
):
    """
    Compress the upper part of a normalized pitch contour.
    Lower/mid pitch stays mostly intact; upper pitch gets bent down.
    """
    pitch_norm = np.asarray(pitch_norm, dtype=np.float32)
    out = pitch_norm.copy()

    zone = np.clip(
        (pitch_norm - threshold) / max(1e-8, (1.0 - threshold)),
        0.0,
        1.0,
    )
    zone_shaped = zone ** power

    compressed_zone = zone * (1.0 - strength * zone_shaped)
    out = np.where(
        pitch_norm > threshold,
        threshold + compressed_zone * (1.0 - threshold),
        pitch_norm,
    )
    return np.clip(out, 0.0, 1.0)


def suppress_copeaks(
    loud,
    pitch_norm,
    loud_threshold=0.40,
    pitch_threshold=0.40,
    strength=0.85,
    power=1.5,
):
    loud = np.asarray(loud, dtype=np.float32)
    pitch_norm = np.asarray(pitch_norm, dtype=np.float32)

    pitch_zone = np.clip(
        (pitch_norm - pitch_threshold) / max(1e-8, (1.0 - pitch_threshold)),
        0.0,
        1.0,
    )
    loud_zone = np.clip(
        (loud - loud_threshold) / max(1e-8, (1.0 - loud_threshold)),
        0.0,
        1.0,
    )

    overlap = (pitch_zone * loud_zone) ** power
    suppress = 1.0 - strength * overlap

    out = loud * suppress
    return np.clip(out, 0.0, 1.0)


def soften_high_pitch_loudness(
    loud,
    pitch_norm,
    threshold=0.58,
    strength=0.35,
    power=1.5,
):
    loud = np.asarray(loud, dtype=np.float32)
    pitch_norm = np.asarray(pitch_norm, dtype=np.float32)

    pitch_zone = np.clip(
        (pitch_norm - threshold) / max(1e-8, (1.0 - threshold)),
        0.0,
        1.0,
    )
    pitch_zone = pitch_zone ** power

    suppress = 1.0 - strength * pitch_zone
    return np.clip(loud * suppress, 0.0, 1.0)


def midi_to_hz(m):
    return 440.0 * (2.0 ** ((m - 69.0) / 12.0))


def hz_to_midi(f):
    f = np.asarray(f, dtype=np.float32)
    out = np.full_like(f, np.nan, dtype=np.float32)
    mask = f > 1e-6
    out[mask] = 69.0 + 12.0 * np.log2(f[mask] / 440.0)
    return out


def safe_nanmean(x, default=np.nan):
    x = np.asarray(x, dtype=np.float32)
    finite = np.isfinite(x)
    if np.any(finite):
        return float(np.mean(x[finite]))
    return float(default)


def downsample_feature(x, factor):
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if factor <= 1:
        return x
    n = len(x) // factor
    if n <= 0:
        return x
    x = x[:n * factor]
    return x.reshape(n, factor).mean(axis=1).astype(np.float32)


def downsample_pitch_feature(x, factor):
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if factor <= 1:
        return x
    n = len(x) // factor
    if n <= 0:
        return x

    x = x[:n * factor].reshape(n, factor)
    out = np.empty(n, dtype=np.float32)

    for i in range(n):
        finite = np.isfinite(x[i])
        if np.any(finite):
            out[i] = np.median(x[i][finite])
        else:
            out[i] = np.nan

    return out


def infer_pitch_array_to_midi(pitch_arr):
    pitch_arr = np.asarray(pitch_arr, dtype=np.float32).reshape(-1)
    finite = pitch_arr[np.isfinite(pitch_arr)]
    if len(finite) == 0:
        return np.full_like(pitch_arr, np.nan, dtype=np.float32)

    med = float(np.nanmedian(finite))
    if med > 120.0:
        return hz_to_midi(pitch_arr)

    return pitch_arr.astype(np.float32)


def scale_pitch_classes(scale_name):
    if scale_name == "C_major":
        return [0, 2, 4, 5, 7, 9, 11]
    if scale_name == "A_minor":
        return [0, 2, 3, 5, 7, 8, 10]
    raise ValueError(f"Unsupported scale: {scale_name}")


def quantize_midi_to_scale(
    midi_value,
    root_midi=60,
    scale_name="C_major",
    pitch_min=55,
    pitch_max=79,
):
    pcs = scale_pitch_classes(scale_name)
    candidates = []
    for m in range(pitch_min, pitch_max + 1):
        rel_pc = (m - root_midi) % 12
        if rel_pc in pcs:
            candidates.append(m)

    if not candidates:
        return int(np.clip(round(midi_value), pitch_min, pitch_max))

    candidates = np.asarray(candidates, dtype=np.float32)
    idx = np.argmin(np.abs(candidates - midi_value))
    return int(candidates[idx])


def curve_to_control_frames(
    smooth_df,
    duration_sec,
    frame_rate,
    source_col,
    smooth_window,
    invert=False,
):
    required_cols = {"x_px", source_col}
    missing = required_cols - set(smooth_df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    x = smooth_df["x_px"].to_numpy(dtype=np.float32)
    y = smooth_df[source_col].to_numpy(dtype=np.float32)

    idx = np.argsort(x)
    x = x[idx]
    y = y[idx]

    if invert:
        y = y.max() - y

    y_norm = normalize_01(y)

    n_frames = int(duration_sec * frame_rate)
    t_src = np.linspace(0, duration_sec, len(y_norm), endpoint=False)
    t_dst = np.linspace(0, duration_sec, n_frames, endpoint=False)

    out = np.interp(t_dst, t_src, y_norm).astype(np.float32)
    out = moving_average(out, smooth_window)
    out = np.clip(out, 0.0, 1.0)
    return out


def apply_pitch_deadband(midi_float, deadband=0.5):
    midi_float = np.asarray(midi_float, dtype=np.float32)
    if len(midi_float) == 0:
        return midi_float.copy()

    out = midi_float.copy()
    last = float(out[0])

    for i in range(1, len(out)):
        if abs(float(out[i]) - last) < deadband:
            out[i] = last
        else:
            last = float(out[i])

    return out


def hold_pitch_plateaus(midi_frames, hold_frames=16):
    midi_frames = np.asarray(midi_frames, dtype=np.int32)
    out = midi_frames.copy()

    n = len(out)
    i = 0
    while i < n:
        j = min(n, i + hold_frames)
        val = int(round(np.median(out[i:j])))
        out[i:j] = val
        i = j

    return out


def constrain_melodic_jumps(notes, max_jump=3):
    notes = np.asarray(notes, dtype=np.int32).copy()
    for i in range(1, len(notes)):
        diff = int(notes[i] - notes[i - 1])
        if diff > max_jump:
            notes[i] = notes[i - 1] + max_jump
        elif diff < -max_jump:
            notes[i] = notes[i - 1] - max_jump
    return notes


def smooth_note_sequence(notes):
    notes = np.asarray(notes, dtype=np.int32).copy()
    if len(notes) < 3:
        return notes

    out = notes.copy()
    for i in range(1, len(notes) - 1):
        if notes[i - 1] == notes[i + 1] and notes[i] != notes[i - 1]:
            out[i] = notes[i - 1]
    return out


def smooth_event_midi_sequence(notes, win=5, blend=0.7):
    notes = np.asarray(notes, dtype=np.float32)
    if len(notes) == 0:
        return notes.astype(np.int32)

    sm = moving_average(notes, win)
    blended = (1.0 - blend) * notes + blend * sm
    return np.rint(blended).astype(np.int32)


def build_pitch_and_loudness_controls(smooth_df):
    # Raw pitch contour
    pitch_base_raw = curve_to_control_frames(
        smooth_df=smooth_df,
        duration_sec=DURATION_SEC,
        frame_rate=FRAME_RATE,
        source_col=PITCH_SOURCE,
        smooth_window=PITCH_SMOOTH_WINDOW,
        invert=PITCH_INVERT,
    )

    # Stronger upper pitch compression
    pitch_base_soft = pitch_base_raw.copy()
    if ENABLE_PITCH_SOFT_CEILING:
        pitch_base_soft = soft_limit_pitch_norm(
            pitch_norm=pitch_base_soft,
            threshold=PITCH_SOFT_CEILING_THRESHOLD,
            strength=PITCH_SOFT_CEILING_STRENGTH,
            power=PITCH_SOFT_CEILING_POWER,
        )

    loud_base_raw = curve_to_control_frames(
        smooth_df=smooth_df,
        duration_sec=DURATION_SEC,
        frame_rate=FRAME_RATE,
        source_col=LOUDNESS_SOURCE,
        smooth_window=LOUD_SMOOTH_WINDOW,
        invert=LOUDNESS_INVERT,
    )

    loud_base = boost_dynamics_np(
        loud_base_raw,
        threshold=EXPAND_THRESHOLD,
        ratio=EXPAND_RATIO,
    )

    diff = np.diff(loud_base, prepend=loud_base[0])
    attack = np.clip(diff, 0.0, None)
    if np.max(attack) > 0:
        attack = attack / (np.max(attack) + 1e-8)

    loud_before_suppression = (1.0 - ATTACK_BLEND) * loud_base + ATTACK_BLEND * attack
    loud_before_suppression = np.clip(loud_before_suppression, 0.0, 1.0)
    loud_before_suppression = LOUD_FLOOR + LOUD_RANGE * loud_before_suppression
    loud_before_suppression = np.clip(loud_before_suppression, 0.0, 1.0)

    loud_after_suppression = loud_before_suppression.copy()

    if ENABLE_COPEAK_SUPPRESSION:
        loud_after_suppression = suppress_copeaks(
            loud=loud_after_suppression,
            pitch_norm=pitch_base_soft,
            loud_threshold=COPEAK_LOUD_THRESHOLD,
            pitch_threshold=COPEAK_PITCH_THRESHOLD,
            strength=COPEAK_SUPPRESS_STRENGTH,
            power=COPEAK_SUPPRESS_POWER,
        )

    if ENABLE_HIGH_PITCH_LOUDNESS_SOFTEN:
        loud_after_suppression = soften_high_pitch_loudness(
            loud=loud_after_suppression,
            pitch_norm=pitch_base_soft,
            threshold=HIGH_PITCH_LOUDNESS_THRESHOLD,
            strength=HIGH_PITCH_LOUDNESS_STRENGTH,
            power=HIGH_PITCH_LOUDNESS_POWER,
        )

    midi_float_raw = PITCH_MIN_MIDI + pitch_base_raw * (PITCH_MAX_MIDI - PITCH_MIN_MIDI)
    midi_float_raw = moving_average(midi_float_raw, MIDI_FLOAT_SMOOTH_WINDOW)
    midi_float_raw = apply_pitch_deadband(midi_float_raw, deadband=MIDI_DEADBAND_SEMITONES)

    midi_float_soft = PITCH_MIN_MIDI + pitch_base_soft * (PITCH_MAX_MIDI - PITCH_MIN_MIDI)
    midi_float_soft = moving_average(midi_float_soft, MIDI_FLOAT_SMOOTH_WINDOW)
    midi_float_soft = apply_pitch_deadband(midi_float_soft, deadband=MIDI_DEADBAND_SEMITONES)

    midi_quant = np.array([
        quantize_midi_to_scale(
            m,
            root_midi=ROOT_MIDI,
            scale_name=SCALE_NAME,
            pitch_min=PITCH_MIN_MIDI,
            pitch_max=PITCH_MAX_MIDI,
        )
        for m in midi_float_soft
    ], dtype=np.int32)

    midi_quant = hold_pitch_plateaus(midi_quant, hold_frames=PLATEAU_HOLD_FRAMES)

    if ENABLE_FINAL_MIDI_CAP:
        midi_quant = np.minimum(midi_quant, FINAL_MIDI_CAP)

    debug = {
        "pitch_base_raw": pitch_base_raw.astype(np.float32),
        "pitch_base_soft": pitch_base_soft.astype(np.float32),
        "midi_float_raw": midi_float_raw.astype(np.float32),
        "midi_float_soft": midi_float_soft.astype(np.float32),
        "loud_before": loud_before_suppression.astype(np.float32),
        "loud_after": loud_after_suppression.astype(np.float32),
    }

    return (
        midi_quant.astype(np.int32),
        loud_after_suppression.astype(np.float32),
        pitch_base_soft.astype(np.float32),
        midi_float_soft.astype(np.float32),
        debug,
    )


def build_note_events_from_line(
    midi_frames,
    loud_frames,
    frame_rate=250,
    min_note_dur_sec=0.20,
    max_note_dur_sec=0.50,
    pitch_change_thresh=1.5,
    loud_attack_thresh=0.10,
):
    midi_frames = np.asarray(midi_frames, dtype=np.float32)
    loud_frames = np.asarray(loud_frames, dtype=np.float32)

    min_frames = max(1, int(round(min_note_dur_sec * frame_rate)))
    max_frames = max(min_frames + 1, int(round(max_note_dur_sec * frame_rate)))

    loud_diff = np.diff(loud_frames, prepend=loud_frames[0])

    boundaries = [0]
    start = 0

    for t in range(1, len(midi_frames)):
        seg_len = t - start
        seg_pitch_med = float(np.median(midi_frames[start:t])) if t > start else float(midi_frames[t])

        pitch_move = abs(float(midi_frames[t]) - seg_pitch_med)
        loud_attack = float(loud_diff[t])

        should_cut = False

        if seg_len >= min_frames:
            if pitch_move >= pitch_change_thresh:
                should_cut = True
            elif loud_attack >= loud_attack_thresh:
                should_cut = True

        if seg_len >= max_frames:
            should_cut = True

        if should_cut:
            boundaries.append(t)
            start = t

    if boundaries[-1] != len(midi_frames):
        boundaries.append(len(midi_frames))

    events = []
    for i in range(len(boundaries) - 1):
        s = boundaries[i]
        e = boundaries[i + 1]
        if e <= s:
            continue

        midi_val = int(round(np.median(midi_frames[s:e])))
        loud_val = float(np.mean(loud_frames[s:e]))
        start_sec = s / frame_rate
        dur_sec = (e - s) / frame_rate

        events.append({
            "start_frame": s,
            "end_frame": e,
            "start_sec": start_sec,
            "dur_sec": dur_sec,
            "midi": midi_val,
            "loudness": loud_val,
        })

    if events:
        corrected = np.array([ev["midi"] for ev in events], dtype=np.int32)

        corrected = smooth_event_midi_sequence(
            corrected,
            win=EVENT_MIDI_SMOOTH_WINDOW,
            blend=EVENT_MIDI_SMOOTH_BLEND,
        )

        corrected = np.array([
            quantize_midi_to_scale(
                m,
                root_midi=ROOT_MIDI,
                scale_name=SCALE_NAME,
                pitch_min=PITCH_MIN_MIDI,
                pitch_max=PITCH_MAX_MIDI,
            )
            for m in corrected
        ], dtype=np.int32)

        corrected = constrain_melodic_jumps(corrected, max_jump=MAX_MELODIC_JUMP)
        corrected = smooth_note_sequence(corrected)

        if ENABLE_FINAL_MIDI_CAP:
            corrected = np.minimum(corrected, FINAL_MIDI_CAP)

        for ev, m in zip(events, corrected):
            ev["midi"] = int(m)

        for i in range(1, len(events)):
            prev = events[i - 1]["midi"]
            curr = events[i]["midi"]
            if abs(curr - prev) == 1 and events[i]["dur_sec"] < 0.28:
                events[i]["midi"] = prev

    if MAX_NOTES is not None:
        events = events[:MAX_NOTES]

    return events


def choose_best_pipe_chunk_fast(
    target_midi,
    target_loudness,
    req_frames,
    pipe_pitch_midi,
    pipe_loud,
    coarse_step_frames=64,
    refine_radius=64,
    refine_step=16,
):
    n = len(pipe_loud)
    req_frames = max(4, int(req_frames))

    if n < req_frames:
        return 0, req_frames, 9999.0

    def score_window(start):
        p_chunk = pipe_pitch_midi[start:start + req_frames]
        l_chunk = pipe_loud[start:start + req_frames]

        finite = np.isfinite(p_chunk)
        finite_count = int(np.sum(finite))

        if finite_count < max(MIN_VALID_PITCH_FRAMES, req_frames // 4):
            return 1e18

        pitch_mean = float(np.mean(p_chunk[finite]))
        pitch_std = float(np.std(p_chunk[finite]))
        loud_mean = safe_nanmean(l_chunk, default=0.0)

        if pitch_std > MAX_PITCH_STD:
            return 1e18

        pitch_cost = abs(pitch_mean - target_midi) * PITCH_WEIGHT
        loud_cost = abs(loud_mean - target_loudness) * LOUD_WEIGHT
        stability_cost = STABILITY_WEIGHT * pitch_std

        return pitch_cost + loud_cost + stability_cost

    coarse_best_start = 0
    coarse_best_score = 1e18

    for start in range(0, n - req_frames + 1, coarse_step_frames):
        s = score_window(start)
        if s < coarse_best_score:
            coarse_best_score = s
            coarse_best_start = start

    lo = max(0, coarse_best_start - refine_radius)
    hi = min(n - req_frames, coarse_best_start + refine_radius)

    best_start = coarse_best_start
    best_score = coarse_best_score

    for start in range(lo, hi + 1, refine_step):
        s = score_window(start)
        if s < best_score:
            best_score = s
            best_start = start

    return best_start, req_frames, best_score


def get_chunk_mean_midi(pipe_pitch_midi, start_frame, n_frames):
    chunk_pitch = pipe_pitch_midi[start_frame:start_frame + n_frames]
    finite = np.isfinite(chunk_pitch)
    if np.any(finite):
        return float(np.mean(chunk_pitch[finite]))
    return np.nan


def slice_or_wrap_audio(audio_flat, start_sample, req_len):
    audio_flat = np.asarray(audio_flat, dtype=np.float32).reshape(-1)
    n = len(audio_flat)

    if req_len <= 0:
        return np.zeros(0, dtype=np.float32)

    if not USE_WRAP:
        end = min(start_sample + req_len, n)
        out = audio_flat[start_sample:end]
        if len(out) < req_len:
            out = np.pad(out, (0, req_len - len(out)))
        return out.astype(np.float32)

    idx = (np.arange(req_len) + start_sample) % n
    return audio_flat[idx].astype(np.float32)


def pitch_shift_chunk_to_target(chunk, sr, source_midi, target_midi):
    chunk = np.asarray(chunk, dtype=np.float32)

    if not np.isfinite(source_midi):
        return chunk

    n_steps = float(target_midi - source_midi)

    if PITCH_SHIFT_CLAMP:
        n_steps = float(np.clip(n_steps, -MAX_SHIFT_SEMITONES, MAX_SHIFT_SEMITONES))

    if abs(n_steps) < 0.12:
        return chunk

    shifted = librosa.effects.pitch_shift(
        y=chunk.astype(np.float32),
        sr=sr,
        n_steps=n_steps,
    )
    return shifted.astype(np.float32)


def apply_fade(wave, fade_ms=10, sr=16000):
    wave = np.asarray(wave, dtype=np.float32).copy()
    fade_n = int(sr * fade_ms / 1000.0)
    fade_n = min(fade_n, len(wave) // 2)
    if fade_n <= 1:
        return wave
    fade_in = np.linspace(0.0, 1.0, fade_n, dtype=np.float32)
    fade_out = np.linspace(1.0, 0.0, fade_n, dtype=np.float32)
    wave[:fade_n] *= fade_in
    wave[-fade_n:] *= fade_out
    return wave


def overlap_add(base, chunk, start_sample):
    end_sample = min(len(base), start_sample + len(chunk))
    if end_sample <= start_sample:
        return base
    base[start_sample:end_sample] += chunk[:end_sample - start_sample]
    return base


def apply_frame_loudness_to_audio(audio_wave, loudness_frames, hop_samples=64):
    audio_wave = np.asarray(audio_wave, dtype=np.float32).reshape(-1)
    loudness_frames = np.asarray(loudness_frames, dtype=np.float32).reshape(-1)

    env = np.repeat(loudness_frames, hop_samples)
    if len(env) < len(audio_wave):
        env = np.pad(env, (0, len(audio_wave) - len(env)), mode="edge")
    else:
        env = env[:len(audio_wave)]

    return (audio_wave * env).astype(np.float32)


def save_control_plot(debug_dict, midi_frames, out_path):
    fig, axes = plt.subplots(4, 1, figsize=(13, 11), sharex=True)

    # 1. Raw vs softened normalized pitch
    axes[0].plot(debug_dict["pitch_base_raw"], linewidth=1.4, label="pitch raw")
    axes[0].plot(debug_dict["pitch_base_soft"], linewidth=1.4, label="pitch softened")
    axes[0].set_title(f"Normalized pitch control ({PITCH_SOURCE})")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    # 2. Raw vs softened continuous MIDI
    axes[1].plot(debug_dict["midi_float_raw"], linewidth=1.4, label="midi float raw")
    axes[1].plot(debug_dict["midi_float_soft"], linewidth=1.4, label="midi float softened")
    axes[1].set_title("Continuous MIDI contour: raw vs softened")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    # 3. Final quantized/capped melody
    axes[2].plot(midi_frames, linewidth=1.4, label="final midi quantized/capped")
    if ENABLE_FINAL_MIDI_CAP:
        axes[2].axhline(FINAL_MIDI_CAP, linestyle="--", linewidth=1.0, label=f"final cap {FINAL_MIDI_CAP}")
    axes[2].set_title(f"Final MIDI melody ({SCALE_NAME})")
    axes[2].grid(alpha=0.3)
    axes[2].legend()

    # 4. Loudness before vs after suppression
    axes[3].plot(debug_dict["loud_before"], linewidth=1.4, label="loud before suppression")
    axes[3].plot(debug_dict["loud_after"], linewidth=1.4, label="loud after suppression")
    axes[3].set_title(f"Loudness control ({LOUDNESS_SOURCE})")
    axes[3].grid(alpha=0.3)
    axes[3].legend()
    axes[3].set_xlabel("Frame")

    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def save_match_plot(note_df, out_path):
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    axes[0].plot(note_df["start_sec"], note_df["midi"], marker="o", linewidth=1.2)
    axes[0].set_title("Line-guided decoded note melody")
    axes[0].set_ylabel("MIDI")
    axes[0].grid(alpha=0.3)

    axes[1].plot(note_df["start_sec"], note_df["match_score"], marker="o", linewidth=1.2)
    axes[1].set_title("Pipe chunk match score per note")
    axes[1].set_ylabel("Score")
    axes[1].set_xlabel("Time (sec)")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# -------------------------------------------------
# MAIN
# -------------------------------------------------
def main():
    print(f"Repo root: {REPO_ROOT}")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Curve CSV: {SMOOTH_CSV}")
    print(f"Pipe audio path: {PIPE_AUDIO_PATH}")
    print(f"Pipe loudness path: {PIPE_LOUD_PATH}")
    print(f"Pipe pitch path: {PIPE_PITCH_PATH}")

    print(f"ENABLE_PITCH_SOFT_CEILING: {ENABLE_PITCH_SOFT_CEILING}")
    print(f"PITCH_SOFT_CEILING_THRESHOLD: {PITCH_SOFT_CEILING_THRESHOLD}")
    print(f"PITCH_SOFT_CEILING_STRENGTH: {PITCH_SOFT_CEILING_STRENGTH}")
    print(f"PITCH_SOFT_CEILING_POWER: {PITCH_SOFT_CEILING_POWER}")
    print(f"ENABLE_FINAL_MIDI_CAP: {ENABLE_FINAL_MIDI_CAP}")
    print(f"FINAL_MIDI_CAP: {FINAL_MIDI_CAP}")

    print(f"ENABLE_COPEAK_SUPPRESSION: {ENABLE_COPEAK_SUPPRESSION}")
    print(f"COPEAK_PITCH_THRESHOLD: {COPEAK_PITCH_THRESHOLD}")
    print(f"COPEAK_LOUD_THRESHOLD: {COPEAK_LOUD_THRESHOLD}")
    print(f"COPEAK_SUPPRESS_STRENGTH: {COPEAK_SUPPRESS_STRENGTH}")
    print(f"COPEAK_SUPPRESS_POWER: {COPEAK_SUPPRESS_POWER}")

    print(f"ENABLE_HIGH_PITCH_LOUDNESS_SOFTEN: {ENABLE_HIGH_PITCH_LOUDNESS_SOFTEN}")
    print(f"HIGH_PITCH_LOUDNESS_THRESHOLD: {HIGH_PITCH_LOUDNESS_THRESHOLD}")
    print(f"HIGH_PITCH_LOUDNESS_STRENGTH: {HIGH_PITCH_LOUDNESS_STRENGTH}")
    print(f"HIGH_PITCH_LOUDNESS_POWER: {HIGH_PITCH_LOUDNESS_POWER}")

    print(f"ATTACK_BLEND: {ATTACK_BLEND}")
    print(f"MAX_SHIFT_SEMITONES: {MAX_SHIFT_SEMITONES}")
    print(f"MAX_MELODIC_JUMP: {MAX_MELODIC_JUMP}")

    if not SMOOTH_CSV.exists():
        raise FileNotFoundError(f"Missing smooth curve CSV: {SMOOTH_CSV}")
    if not PIPE_AUDIO_PATH.exists():
        raise FileNotFoundError(f"Missing pipe audio npy: {PIPE_AUDIO_PATH}")
    if not PIPE_LOUD_PATH.exists():
        raise FileNotFoundError(f"Missing pipe loudness npy: {PIPE_LOUD_PATH}")
    if not PIPE_PITCH_PATH.exists():
        raise FileNotFoundError(f"Missing pipe pitch npy: {PIPE_PITCH_PATH}")

    smooth_df = pd.read_csv(SMOOTH_CSV)
    print("Loaded curve columns:", smooth_df.columns.tolist())
    print("Number of curve rows:", len(smooth_df))

    midi_frames, loudness_frames, pitch_curve, midi_float, debug = build_pitch_and_loudness_controls(smooth_df)

    pipe_audio_flat = flatten_feature_array(np.load(PIPE_AUDIO_PATH, mmap_mode="r"))
    pipe_loud_flat = flatten_feature_array(np.load(PIPE_LOUD_PATH, mmap_mode="r"))
    pipe_pitch_raw = flatten_feature_array(np.load(PIPE_PITCH_PATH, mmap_mode="r"))
    pipe_pitch_midi = infer_pitch_array_to_midi(pipe_pitch_raw)

    min_frames = min(len(pipe_loud_flat), len(pipe_pitch_midi))
    pipe_loud_flat = pipe_loud_flat[:min_frames]
    pipe_pitch_midi = pipe_pitch_midi[:min_frames]

    pipe_loud_search = downsample_feature(pipe_loud_flat, SEARCH_DS)
    pipe_pitch_search = downsample_pitch_feature(pipe_pitch_midi, SEARCH_DS)

    valid_pitch_ratio = float(np.mean(np.isfinite(pipe_pitch_midi)))
    print("Full pipe audio shape:", pipe_audio_flat.shape)
    print("Full pipe loudness shape:", pipe_loud_flat.shape)
    print("Full pipe pitch shape:", pipe_pitch_midi.shape)
    print("Search loudness shape:", pipe_loud_search.shape)
    print("Search pitch shape:", pipe_pitch_search.shape)
    print(f"Valid pitch ratio: {valid_pitch_ratio:.4f}")
    print(f"SEARCH_DS: {SEARCH_DS}")

    if valid_pitch_ratio < 0.01:
        raise ValueError("Pipe pitch file appears mostly invalid. Retrieval would be nonsense.")

    events = build_note_events_from_line(
        midi_frames,
        loudness_frames,
        frame_rate=FRAME_RATE,
        min_note_dur_sec=MIN_NOTE_DUR_SEC,
        max_note_dur_sec=MAX_NOTE_DUR_SEC,
        pitch_change_thresh=PITCH_CHANGE_THRESH,
        loud_attack_thresh=LOUD_ATTACK_THRESH,
    )

    print(f"Number of note events: {len(events)}")
    if events:
        print(f"First event start: {events[0]['start_sec']:.2f}s")
        print(f"Last event start:  {events[-1]['start_sec']:.2f}s")
        print(f"Last event end:    {events[-1]['start_sec'] + events[-1]['dur_sec']:.2f}s")

    total_samples = int(round(DURATION_SEC * SR))
    out_audio = np.zeros(total_samples, dtype=np.float32)
    rows = []

    for k, ev in enumerate(events):
        req_frames = max(
            int(round(0.24 * FRAME_RATE)),
            int(round(ev["dur_sec"] * FRAME_RATE)),
        ) + CHUNK_PAD_FRAMES

        req_frames_search = max(4, req_frames // SEARCH_DS)

        coarse_step_search = max(4, COARSE_STEP_FRAMES // SEARCH_DS)
        refine_radius_search = max(8, REFINE_RADIUS_FRAMES // SEARCH_DS)
        refine_step_search = max(2, REFINE_STEP_FRAMES // SEARCH_DS)

        best_start_search, best_len_search, best_score = choose_best_pipe_chunk_fast(
            target_midi=ev["midi"],
            target_loudness=ev["loudness"],
            req_frames=req_frames_search,
            pipe_pitch_midi=pipe_pitch_search,
            pipe_loud=pipe_loud_search,
            coarse_step_frames=coarse_step_search,
            refine_radius=refine_radius_search,
            refine_step=refine_step_search,
        )

        best_start_frame = best_start_search * SEARCH_DS
        best_len_frames = req_frames

        start_sample_mem = best_start_frame * HOP_SAMPLES
        req_samples = best_len_frames * HOP_SAMPLES

        chunk = slice_or_wrap_audio(
            pipe_audio_flat,
            start_sample=start_sample_mem,
            req_len=req_samples,
        )

        source_midi = get_chunk_mean_midi(
            pipe_pitch_midi=pipe_pitch_midi,
            start_frame=best_start_frame,
            n_frames=best_len_frames,
        )

        if ENABLE_PITCH_SHIFT:
            chunk = pitch_shift_chunk_to_target(
                chunk=chunk,
                sr=SR,
                source_midi=source_midi,
                target_midi=ev["midi"],
            )

        peak = np.max(np.abs(chunk))
        if peak > 1e-8:
            chunk = chunk / peak

        chunk = chunk * float(ev["loudness"])
        chunk = apply_fade(chunk, fade_ms=FADE_MS, sr=SR)

        out_start = int(round(ev["start_sec"] * SR))
        out_audio = overlap_add(out_audio, chunk, out_start)

        shift_semitones = float(ev["midi"] - source_midi) if np.isfinite(source_midi) else np.nan

        rows.append({
            "note_idx": k,
            "start_sec": ev["start_sec"],
            "dur_sec": ev["dur_sec"],
            "midi": ev["midi"],
            "freq_hz": midi_to_hz(ev["midi"]),
            "loudness": ev["loudness"],
            "match_start_frame": int(best_start_frame),
            "match_score": float(best_score),
            "source_midi": float(source_midi) if np.isfinite(source_midi) else np.nan,
            "shift_semitones": shift_semitones,
        })

        src_txt = f"{source_midi:.2f}" if np.isfinite(source_midi) else "nan"
        shift_txt = f"{shift_semitones:.2f}" if np.isfinite(shift_semitones) else "nan"

        print(
            f"Rendered note {k+1}/{len(events)} | "
            f"target_midi={ev['midi']} | src_midi={src_txt} | "
            f"shift={shift_txt} st | loud={ev['loudness']:.3f} | "
            f"search_start={best_start_search} | score={best_score:.3f}"
        )

    out_audio = apply_frame_loudness_to_audio(
        out_audio,
        loudness_frames=loudness_frames,
        hop_samples=HOP_SAMPLES,
    )

    out_audio *= PIPE_VOLUME
    peak = np.max(np.abs(out_audio))
    if peak > 1e-8:
        out_audio = out_audio / peak * 0.95

    nz = np.where(np.abs(out_audio) > 1e-5)[0]
    if len(nz) > 0:
        print(f"Audio energy from {nz[0]/SR:.2f}s to {nz[-1]/SR:.2f}s")
    else:
        print("Output audio is effectively silent.")

    note_df = pd.DataFrame(rows)

    sf.write(OUT_WAV, out_audio.astype(np.float32), SR)
    note_df.to_csv(OUT_NOTE_CSV, index=False)
    save_control_plot(debug, midi_frames, OUT_CTRL_PLOT)
    save_match_plot(note_df, OUT_MATCH_PLOT)

    print("Done.")
    print(f"Saved WAV: {OUT_WAV}")
    print(f"Saved notes CSV: {OUT_NOTE_CSV}")
    print(f"Saved control plot: {OUT_CTRL_PLOT}")
    print(f"Saved match plot: {OUT_MATCH_PLOT}")


if __name__ == "__main__":
    main()