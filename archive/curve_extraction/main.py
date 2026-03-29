from curve_utils import run_pipeline
from audio_utils import sonify_curve_as_guitar
from report_utils import save_guitar_report_card

# =========================
# USER CONFIG
# =========================
IMAGE_PATH = "input/img56.png"
OUTPUT_DIR = "output"

# Baseline choice
ZERO_MODE = "horizontal"
ZERO_LINE_Y = 2000

# Only used if ZERO_MODE != "horizontal"
ZERO_P1 = (0, 240)
ZERO_P2 = (1000, 240)

# Preprocessing
PREPROCESS_MODE = "canny"  # try "binary" too for skyline images
GAUSSIAN_KERNEL = 9
CANNY_LOW = 80
CANNY_HIGH = 190

# Boundary extraction
BOUNDARY_MODE = "upper"     # "upper" or "generic"
USE_SKELETON = False        # only matters for generic mode
MIN_CURVE_POINTS = 30
MIN_RUN = 1
SPIKE_MAX_JUMP = 15

# Smoothing / simplification
SMOOTH_WINDOW = 25
SMOOTH_METHOD = "savgol"
SMOOTH_POLYORDER = 3
SIMPLIFY_STEP = 10

# Signal analysis
PEAK_PROMINENCE = 9.0


def main():
    raw_df, smooth_df, simplified_df, events_df = run_pipeline(
        image_path=IMAGE_PATH,
        output_dir=OUTPUT_DIR,
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

    x_curve = smooth_df["x_px"].to_numpy()
    y_curve = smooth_df["displacement_px"].to_numpy()

    audio, env, pitch_hz, sr = sonify_curve_as_guitar(
        x_curve,
        y_curve,
        out_wav=f"{OUTPUT_DIR}/guitar_curve.wav",
        out_env_npy=f"{OUTPUT_DIR}/guitar_curve_env.npy",
        out_pitch_npy=f"{OUTPUT_DIR}/guitar_curve_pitch.npy",
        sr=16000,
        duration=20.0,
        smooth_window=101,
        fmin=82.41,
        fmax=659.25,
        quantize_scale=True,
    )

    save_guitar_report_card(
        f"{OUTPUT_DIR}/guitar_curve.wav",
        env_npy_path=f"{OUTPUT_DIR}/guitar_curve_env.npy",
        pitch_npy_path=f"{OUTPUT_DIR}/guitar_curve_pitch.npy",
    )

    print("Done.")
    print(f"Audio samples: {len(audio)}")
    print(f"Envelope samples: {len(env)}")
    print(f"Pitch samples: {len(pitch_hz)}")
    print(f"Saved: {OUTPUT_DIR}/guitar_curve.wav")
    print(f"Saved: {OUTPUT_DIR}/guitar_curve_env.npy")
    print(f"Saved: {OUTPUT_DIR}/guitar_curve_pitch.npy")
    print(f"Saved: {OUTPUT_DIR}/guitar_curve.png")


if __name__ == "__main__":
    main()