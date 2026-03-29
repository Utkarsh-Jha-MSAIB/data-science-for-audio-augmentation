import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import find_peaks, savgol_filter
from skimage.morphology import skeletonize


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_image(image_path: str) -> np.ndarray:
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return image


def preprocess_image(
    image_bgr: np.ndarray,
    mode: str = "canny",
    gaussian_kernel: int = 5,
    canny_low: int = 50,
    canny_high: int = 150,
) -> tuple[np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    if gaussian_kernel % 2 == 0:
        gaussian_kernel += 1

    blurred = cv2.GaussianBlur(gray, (gaussian_kernel, gaussian_kernel), 0)

    if mode == "binary":
        _, mask = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
    else:
        mask = cv2.Canny(blurred, canny_low, canny_high)

    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    return gray, cleaned


def extract_curve_points(
    mask: np.ndarray,
    use_skeleton: bool = True,
    min_curve_points: int = 30,
) -> np.ndarray:
    if use_skeleton:
        binary = (mask > 0).astype(np.uint8)
        skel = skeletonize(binary).astype(np.uint8)
        ys, xs = np.where(skel > 0)
        if len(xs) == 0:
            raise ValueError("Skeleton produced no curve points.")
        return np.column_stack([xs, ys]).astype(np.int32)

    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    valid = [c for c in contours if len(c) >= min_curve_points]
    if not valid:
        raise ValueError("No valid contour found.")
    contour = max(valid, key=lambda c: len(c))
    return contour[:, 0, :]


def extract_upper_boundary_from_mask(mask: np.ndarray, min_run: int = 3) -> np.ndarray:
    """
    For each x-column, find the first foreground pixel from the top.
    """
    h, w = mask.shape
    xs = []
    ys = []

    for x in range(w):
        col = mask[:, x] > 0
        y_found = None

        for y in range(h - min_run + 1):
            if np.all(col[y:y + min_run]):
                y_found = y
                break

        if y_found is not None:
            xs.append(x)
            ys.append(y_found)

    if len(xs) == 0:
        raise ValueError("No upper boundary found.")

    return np.column_stack([np.array(xs), np.array(ys)]).astype(np.int32)


def extract_top_bottom_height_profile_from_mask(
    mask: np.ndarray,
    min_run: int = 3,
) -> pd.DataFrame:
    """
    For each x-column, find:
      - top foreground y
      - bottom foreground y
      - height = bottom - top
    This captures silhouette thickness/height per column.
    """
    h, w = mask.shape
    rows = []

    for x in range(w):
        col = mask[:, x] > 0
        ys = np.where(col)[0]

        if len(ys) < min_run:
            continue

        top = int(ys[0])
        bottom = int(ys[-1])

        rows.append({
            "x_px": float(x),
            "y_top_px": float(top),
            "y_bottom_px": float(bottom),
            "height_px": float(bottom - top),
        })

    if len(rows) == 0:
        raise ValueError("No valid height profile found.")

    return pd.DataFrame(rows)


def points_to_curve_dataframe(points: np.ndarray) -> pd.DataFrame:
    if points.shape[0] == 0:
        raise ValueError("No points found for curve extraction.")

    x_vals = points[:, 0]
    y_vals = points[:, 1]

    xs = np.arange(int(x_vals.min()), int(x_vals.max()) + 1)
    ys = np.full(xs.shape, np.nan, dtype=float)

    for i, x in enumerate(xs):
        candidates = y_vals[x_vals == x]
        if len(candidates) > 0:
            ys[i] = float(np.median(candidates))

    valid = ~np.isnan(ys)
    if valid.sum() < 2:
        raise ValueError("Not enough valid points to interpolate the curve.")

    interp = interp1d(
        xs[valid],
        ys[valid],
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate",
    )
    y_curve = interp(xs)

    return pd.DataFrame({
        "x_px": xs.astype(float),
        "y_curve_px": y_curve.astype(float),
    })


def densify_height_profile(height_df: pd.DataFrame) -> pd.DataFrame:
    x_vals = height_df["x_px"].to_numpy(dtype=float)
    xs = np.arange(int(np.min(x_vals)), int(np.max(x_vals)) + 1)

    out = {"x_px": xs.astype(float)}

    for col in ["y_top_px", "y_bottom_px", "height_px"]:
        y_vals = height_df[col].to_numpy(dtype=float)
        interp = interp1d(
            x_vals,
            y_vals,
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate",
        )
        out[col] = interp(xs).astype(float)

    return pd.DataFrame(out)


def build_baseline(
    x: np.ndarray,
    zero_mode: str = "horizontal",
    zero_line_y: float = 200.0,
    p1: tuple[float, float] = (0.0, 200.0),
    p2: tuple[float, float] = (1000.0, 200.0),
) -> np.ndarray:
    if zero_mode == "horizontal":
        return np.full_like(x, zero_line_y, dtype=float)

    x1, y1 = p1
    x2, y2 = p2

    if abs(x2 - x1) < 1e-9:
        return np.full_like(x, y1, dtype=float)

    m = (y2 - y1) / (x2 - x1)
    return y1 + m * (x - x1)


def suppress_upward_spikes(y: np.ndarray, max_jump: float = 20.0) -> np.ndarray:
    """
    Suppress isolated upward spikes in image space.
    Smaller y means visually higher in the image.
    """
    y = y.copy()

    for i in range(1, len(y) - 1):
        prev_y = y[i - 1]
        curr_y = y[i]
        next_y = y[i + 1]

        local_ref = 0.5 * (prev_y + next_y)
        if local_ref - curr_y > max_jump:
            y[i] = local_ref

    return y


def smooth_signal(
    signal: np.ndarray,
    method: str = "savgol",
    window: int = 21,
    polyorder: int = 3,
) -> np.ndarray:
    if len(signal) < 5:
        return signal.copy()

    if window <= 1:
        return signal.copy()

    if window % 2 == 0:
        window += 1

    max_valid_window = len(signal) if len(signal) % 2 == 1 else len(signal) - 1
    window = min(window, max_valid_window)

    if window < 3:
        return signal.copy()

    if method == "moving_average":
        kernel = np.ones(window) / window
        return np.convolve(signal, kernel, mode="same")

    polyorder = min(polyorder, window - 1)
    return savgol_filter(signal, window_length=window, polyorder=polyorder)


def downsample_curve(curve_df: pd.DataFrame, step: int = 10) -> pd.DataFrame:
    if step <= 1:
        return curve_df.copy()
    return curve_df.iloc[::step, :].reset_index(drop=True)


def detect_peaks_and_dips(
    x: np.ndarray,
    signal: np.ndarray,
    prominence: float = 8.0,
) -> pd.DataFrame:
    peaks, peak_props = find_peaks(signal, prominence=prominence)
    dips, dip_props = find_peaks(-signal, prominence=prominence)

    rows = []

    for i, idx in enumerate(peaks):
        rows.append({
            "type": "peak",
            "x_px": float(x[idx]),
            "amplitude_px": float(signal[idx]),
            "prominence_px": float(peak_props["prominences"][i]),
        })

    for i, idx in enumerate(dips):
        rows.append({
            "type": "dip",
            "x_px": float(x[idx]),
            "amplitude_px": float(signal[idx]),
            "prominence_px": float(dip_props["prominences"][i]),
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("x_px").reset_index(drop=True)
    return df


def save_overlay_image(
    image_bgr: np.ndarray,
    raw_df: pd.DataFrame,
    smooth_df: pd.DataFrame,
    out_path: str,
    zero_mode: str = "horizontal",
    zero_line_y: float = 200.0,
    p1: tuple[float, float] = (0.0, 200.0),
    p2: tuple[float, float] = (1000.0, 200.0),
) -> None:
    overlay = image_bgr.copy()

    for _, row in raw_df.iterrows():
        x = int(round(row["x_px"]))
        y = int(round(row["y_curve_px"]))
        if 0 <= x < overlay.shape[1] and 0 <= y < overlay.shape[0]:
            overlay[y, x] = (0, 0, 255)

    for _, row in smooth_df.iterrows():
        x = int(round(row["x_px"]))
        y = int(round(row["y_curve_px"]))
        if 0 <= x < overlay.shape[1] and 0 <= y < overlay.shape[0]:
            overlay[y, x] = (0, 255, 255)

    if zero_mode == "horizontal":
        y0 = int(round(zero_line_y))
        cv2.line(overlay, (0, y0), (overlay.shape[1] - 1, y0), (0, 255, 0), 2)
    else:
        cv2.line(
            overlay,
            (int(round(p1[0])), int(round(p1[1]))),
            (int(round(p2[0])), int(round(p2[1]))),
            (0, 255, 0),
            2,
        )

    cv2.imwrite(out_path, overlay)


def save_smooth_overlay_image(
    image_bgr: np.ndarray,
    smooth_df: pd.DataFrame,
    out_path: str,
    zero_mode: str = "horizontal",
    zero_line_y: float = 200.0,
    p1: tuple[float, float] = (0.0, 200.0),
    p2: tuple[float, float] = (1000.0, 200.0),
) -> None:
    overlay = image_bgr.copy()

    for _, row in smooth_df.iterrows():
        x = int(round(row["x_px"]))
        y = int(round(row["y_curve_px"]))
        if 0 <= x < overlay.shape[1] and 0 <= y < overlay.shape[0]:
            overlay[y, x] = (0, 255, 255)

    if zero_mode == "horizontal":
        y0 = int(round(zero_line_y))
        cv2.line(overlay, (0, y0), (overlay.shape[1] - 1, y0), (0, 255, 0), 2)
    else:
        cv2.line(
            overlay,
            (int(round(p1[0])), int(round(p1[1]))),
            (int(round(p2[0])), int(round(p2[1]))),
            (0, 255, 0),
            2,
        )

    cv2.imwrite(out_path, overlay)


def save_height_overlay_image(
    image_bgr: np.ndarray,
    height_df: pd.DataFrame,
    out_path: str,
) -> None:
    overlay = image_bgr.copy()

    for _, row in height_df.iterrows():
        x = int(round(row["x_px"]))
        y_top = int(round(row["y_top_px"]))
        y_bottom = int(round(row["y_bottom_px"]))

        if 0 <= x < overlay.shape[1]:
            if 0 <= y_top < overlay.shape[0]:
                overlay[y_top, x] = (0, 255, 255)
            if 0 <= y_bottom < overlay.shape[0]:
                overlay[y_bottom, x] = (255, 0, 255)

            y1 = max(0, min(overlay.shape[0] - 1, y_top))
            y2 = max(0, min(overlay.shape[0] - 1, y_bottom))
            cv2.line(overlay, (x, y1), (x, y2), (255, 255, 0), 1)

    cv2.imwrite(out_path, overlay)


def save_signal_plot(
    raw_df: pd.DataFrame,
    smooth_df: pd.DataFrame,
    events_df: pd.DataFrame,
    out_path: str,
    signal_col: str = "displacement_px",
    ylabel: str = "Displacement (pixels)",
    title: str = "Signal",
) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))

    ax.plot(raw_df["x_px"], raw_df[signal_col], label="Raw")
    ax.plot(smooth_df["x_px"], smooth_df[signal_col], label="Smoothed")

    if not events_df.empty:
        peaks = events_df[events_df["type"] == "peak"]
        dips = events_df[events_df["type"] == "dip"]

        if not peaks.empty:
            py = np.interp(peaks["x_px"], smooth_df["x_px"], smooth_df[signal_col])
            ax.scatter(peaks["x_px"], py, marker="o", label="Peaks")

        if not dips.empty:
            dy = np.interp(dips["x_px"], smooth_df["x_px"], smooth_df[signal_col])
            ax.scatter(dips["x_px"], dy, marker="x", label="Dips")

    ax.axhline(0, linestyle="--")
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_smooth_curve_plot(
    smooth_df: pd.DataFrame,
    events_df: pd.DataFrame,
    out_path: str,
    signal_col: str = "displacement_px",
    ylabel: str = "Smoothed displacement (pixels)",
    title: str = "Smoothed Curve",
) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))

    ax.plot(smooth_df["x_px"], smooth_df[signal_col], linewidth=2, label="Smoothed")

    if not events_df.empty:
        peaks = events_df[events_df["type"] == "peak"]
        dips = events_df[events_df["type"] == "dip"]

        if not peaks.empty:
            py = np.interp(peaks["x_px"], smooth_df["x_px"], smooth_df[signal_col])
            ax.scatter(peaks["x_px"], py, marker="o", label="Peaks")

        if not dips.empty:
            dy = np.interp(dips["x_px"], smooth_df["x_px"], smooth_df[signal_col])
            ax.scatter(dips["x_px"], dy, marker="x", label="Dips")

    ax.axhline(0, linestyle="--")
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def run_pipeline(
    image_path: str,
    output_dir: str,
    zero_mode: str = "horizontal",
    zero_line_y: float = 200.0,
    zero_p1: tuple[float, float] = (0.0, 200.0),
    zero_p2: tuple[float, float] = (1000.0, 200.0),
    preprocess_mode: str = "canny",
    gaussian_kernel: int = 5,
    canny_low: int = 50,
    canny_high: int = 150,
    boundary_mode: str = "upper",   # "upper" or "generic"
    use_skeleton: bool = False,
    min_curve_points: int = 30,
    min_run: int = 3,
    spike_max_jump: float = 18.0,
    smooth_window: int = 31,
    smooth_method: str = "savgol",
    smooth_polyorder: int = 3,
    simplify_step: int = 10,
    peak_prominence: float = 8.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Backward-compatible return:
      raw_df, smooth_df, simplified_df, events_df

    Additional height-profile artifacts are also saved to disk.
    """
    ensure_dir(output_dir)

    image_bgr = load_image(image_path)
    _, mask = preprocess_image(
        image_bgr=image_bgr,
        mode=preprocess_mode,
        gaussian_kernel=gaussian_kernel,
        canny_low=canny_low,
        canny_high=canny_high,
    )

    # -----------------------------------------------------
    # Primary curve extraction
    # -----------------------------------------------------
    if boundary_mode == "upper":
        points = extract_upper_boundary_from_mask(mask, min_run=min_run)
    else:
        points = extract_curve_points(
            mask=mask,
            use_skeleton=use_skeleton,
            min_curve_points=min_curve_points,
        )

    base_curve_df = points_to_curve_dataframe(points)

    x = base_curve_df["x_px"].to_numpy()
    baseline = build_baseline(
        x=x,
        zero_mode=zero_mode,
        zero_line_y=zero_line_y,
        p1=zero_p1,
        p2=zero_p2,
    )

    raw_df = base_curve_df.copy()
    raw_df["y_curve_px"] = suppress_upward_spikes(
        raw_df["y_curve_px"].to_numpy(),
        max_jump=spike_max_jump,
    )
    raw_df["y_baseline_px"] = baseline
    raw_df["displacement_px"] = raw_df["y_baseline_px"] - raw_df["y_curve_px"]

    smooth_y = smooth_signal(
        raw_df["y_curve_px"].to_numpy(),
        method=smooth_method,
        window=smooth_window,
        polyorder=smooth_polyorder,
    )

    smooth_df = raw_df[["x_px", "y_baseline_px"]].copy()
    smooth_df["y_curve_px"] = smooth_y
    smooth_df["displacement_px"] = smooth_df["y_baseline_px"] - smooth_df["y_curve_px"]
    smooth_df = smooth_df[["x_px", "y_curve_px", "y_baseline_px", "displacement_px"]]

    simplified_df = downsample_curve(smooth_df, step=simplify_step)

    events_df = detect_peaks_and_dips(
        x=smooth_df["x_px"].to_numpy(),
        signal=smooth_df["displacement_px"].to_numpy(),
        prominence=peak_prominence,
    )

    # -----------------------------------------------------
    # Secondary height-profile extraction
    # -----------------------------------------------------
    height_base_df = extract_top_bottom_height_profile_from_mask(mask, min_run=min_run)
    height_base_df = densify_height_profile(height_base_df)

    height_raw_df = height_base_df.copy()
    height_raw_df["height_px"] = smooth_signal(
        height_raw_df["height_px"].to_numpy(),
        method="moving_average",
        window=max(5, simplify_step if simplify_step % 2 == 1 else simplify_step + 1),
        polyorder=1,
    )

    height_smooth_df = height_raw_df.copy()
    height_smooth_df["height_px"] = smooth_signal(
        height_raw_df["height_px"].to_numpy(),
        method=smooth_method,
        window=smooth_window,
        polyorder=smooth_polyorder,
    )

    height_simplified_df = downsample_curve(height_smooth_df, step=simplify_step)

    height_events_df = detect_peaks_and_dips(
        x=height_smooth_df["x_px"].to_numpy(),
        signal=height_smooth_df["height_px"].to_numpy(),
        prominence=peak_prominence,
    )

    # -----------------------------------------------------
    # Save CSVs
    # -----------------------------------------------------
    raw_df.to_csv(os.path.join(output_dir, "curve_mapping_raw.csv"), index=False)
    smooth_df.to_csv(os.path.join(output_dir, "curve_mapping_smooth.csv"), index=False)
    simplified_df.to_csv(os.path.join(output_dir, "curve_mapping_simplified.csv"), index=False)
    events_df.to_csv(os.path.join(output_dir, "peaks_dips.csv"), index=False)

    height_raw_df.to_csv(os.path.join(output_dir, "height_profile_raw.csv"), index=False)
    height_smooth_df.to_csv(os.path.join(output_dir, "height_profile_smooth.csv"), index=False)
    height_simplified_df.to_csv(os.path.join(output_dir, "height_profile_simplified.csv"), index=False)
    height_events_df.to_csv(os.path.join(output_dir, "height_peaks_dips.csv"), index=False)

    # -----------------------------------------------------
    # Save overlay images
    # -----------------------------------------------------
    save_overlay_image(
        image_bgr=image_bgr,
        raw_df=raw_df,
        smooth_df=smooth_df,
        out_path=os.path.join(output_dir, "overlay.png"),
        zero_mode=zero_mode,
        zero_line_y=zero_line_y,
        p1=zero_p1,
        p2=zero_p2,
    )

    save_smooth_overlay_image(
        image_bgr=image_bgr,
        smooth_df=smooth_df,
        out_path=os.path.join(output_dir, "overlay_smooth_only.png"),
        zero_mode=zero_mode,
        zero_line_y=zero_line_y,
        p1=zero_p1,
        p2=zero_p2,
    )

    save_height_overlay_image(
        image_bgr=image_bgr,
        height_df=height_smooth_df,
        out_path=os.path.join(output_dir, "height_overlay.png"),
    )

    # -----------------------------------------------------
    # Save plots
    # -----------------------------------------------------
    save_signal_plot(
        raw_df=raw_df,
        smooth_df=smooth_df,
        events_df=events_df,
        out_path=os.path.join(output_dir, "signal_plot.png"),
        signal_col="displacement_px",
        ylabel="Displacement (pixels)",
        title="Curve Displacement Signal",
    )

    save_smooth_curve_plot(
        smooth_df=smooth_df,
        events_df=events_df,
        out_path=os.path.join(output_dir, "smooth_curve_plot.png"),
        signal_col="displacement_px",
        ylabel="Smoothed displacement (pixels)",
        title="Smoothed Curve",
    )

    save_signal_plot(
        raw_df=height_raw_df,
        smooth_df=height_smooth_df,
        events_df=height_events_df,
        out_path=os.path.join(output_dir, "height_signal_plot.png"),
        signal_col="height_px",
        ylabel="Height (pixels)",
        title="Column-wise Height Signal",
    )

    save_smooth_curve_plot(
        smooth_df=height_smooth_df,
        events_df=height_events_df,
        out_path=os.path.join(output_dir, "height_smooth_plot.png"),
        signal_col="height_px",
        ylabel="Smoothed height (pixels)",
        title="Smoothed Height Profile",
    )

    return raw_df, smooth_df, simplified_df, events_df