import json
from pathlib import Path
import pandas as pd
import numpy as np

# ==========================================
# PATHS
# ==========================================
JSON_PATH = Path("analysis_by_file.json")
OUT_DIR = Path("csv_from_analysis_v2")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# HELPERS
# ==========================================
def safe_name(name: str) -> str:
    keep = []
    for ch in name:
        if ch.isalnum() or ch in ("_", "-", "."):
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep)

def pad_to_same_length(times, values, target_len):
    times = list(times)
    values = list(values)

    if len(times) < target_len:
        times = times + [np.nan] * (target_len - len(times))
    if len(values) < target_len:
        values = values + [np.nan] * (target_len - len(values))

    return times, values

# ==========================================
# LOAD JSON
# ==========================================
with open(JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

all_combined = []

# ==========================================
# PROCESS EACH AUDIO FILE
# ==========================================
for wav_name, entry in data.items():
    rms = entry.get("rms", {})
    f0 = entry.get("f0", {})
    scores = entry.get("scores", {})
    duration = entry.get("duration", None)

    rms_t = rms.get("t", [])
    rms_y = rms.get("y", [])

    f0_t = f0.get("t", [])
    f0_y = f0.get("y", [])

    # make same row count so both can sit in one csv
    n = max(len(rms_t), len(rms_y), len(f0_t), len(f0_y))

    rms_t, rms_y = pad_to_same_length(rms_t, rms_y, n)
    f0_t, f0_y = pad_to_same_length(f0_t, f0_y, n)

    df = pd.DataFrame({
        "file": [wav_name] * n,
        "duration_sec": [duration] * n,
        "rms_time": rms_t,
        "loudness_rms_norm": rms_y,
        "f0_time": f0_t,
        "pitch_hz": f0_y,
        "AvgLoudness_100": [scores.get("AvgLoudness_100")] * n,
        "Loudness_CV": [scores.get("Loudness_CV")] * n,
        "MedianPitch_Hz": [scores.get("MedianPitch_Hz")] * n,
    })

    # save one csv per instrument/file
    out_csv = OUT_DIR / f"{safe_name(Path(wav_name).stem)}.csv"
    df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

    all_combined.append(df)

# ==========================================
# SAVE COMBINED CSV
# ==========================================
if all_combined:
    df_all = pd.concat(all_combined, ignore_index=True)
    combined_csv = OUT_DIR / "all_instruments_pitch_loudness.csv"
    df_all.to_csv(combined_csv, index=False)
    print(f"Saved combined CSV: {combined_csv}")