import argparse
import csv
import ctypes
import os
import struct
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from arena_api.system import system
from lens import Lens

# =============================
# 参数（根据你的相机修改）
# =============================
IMG_H = 720
IMG_W = 1280
TIME_WINDOW = 0.05  # 时间窗口（秒）

# =============================
# 0. 采集参数
# =============================
CAPTURE_DURATION_MS = 10000
FRAME_INTERVAL_MS = 50
DIOPTER_UPDATE_MS = 20
START_DIOPTER = -4.0
END_DIOPTER = 4.0
FOCUS_WINDOW_MS = 20
FOCUS_ALIGN_LAG_MS = 0.0
ALIGN_SEARCH_RANGE_MS = 100
ALIGN_SEARCH_STEP_MS = 10
APPLY_ESTIMATED_LAG = False
FOCUS_DETECT_MODE = "argmax"  # "argmax" or "zero_cross"
FILTER_ELP = True
FILTER_THRESHOLD = 1e5
FILTER_ALPHA = 0.3
FILTER_WINDOW = 15
ZERO_CROSS_NEG_THRESHOLD = -1000
ZERO_CROSS_POS_THRESHOLD = 100
RECORD_ALL_TRANSITIONS = True
TRANSITION_USE_THRESH = True
TRANSITION_MIN_POS = ZERO_CROSS_POS_THRESHOLD
TRANSITION_MAX_NEG = ZERO_CROSS_NEG_THRESHOLD
TRANSITION_SAVE_IMAGES = False
TRANSITION_IMAGE_DIR = "focus_transitions"
RECORD_JUMPS = True
TRANSITION_JUMP_THRESHOLD = 40.0
MIN_SEARCH_MS = 500
SELECT_TRANSITION_BY_SHARPNESS = True
TRANSITION_WINDOW_MS = 20
MIN_TRANSITION_EVENTS = 0
MAX_TRANSITION_WINDOW_MULT = 1
MIN_FOCUS_EVENTS = 0
MAX_FOCUS_WINDOW_MULT = 1
CORR_SG_WINDOW = 11
CORR_SG_POLYORDER = 2
CORR_PEAK_MIN_SCORE = 0.0
CORR_PEAK_MIN_RATIO = 0.4
CORR_PEAK_MIN_EVENTS = 1000
CORR_TREND_WINDOW = 3
USE_ROI = False
ROI = (0, 0, 0, 0)  # (y1, y2, x1, x2)
SAVE_FOCUS_GRAY = False
EVENT_RENDER_PERCENTILE = 90
EVENT_RENDER_GAMMA = 0.6
EVENT_RENDER_DENOISE = False
ENABLE_BIAS_PRESET = True
BIAS_DELTAS = {
    # Increase thresholds slightly to reduce background activity
    "BiasDiffOn": 2,
    "BiasDiffOff": 2,
     "BiasRefractory": 2,
    "BiasHpf": 2,
    "BiasFo": 2,
    "EventThreshold": 1,
    "EventThresholdOn": 1,
    "EventThresholdOff": 1,
}
CSV_DIR = r"C:\Users\13119\OneDrive\Desktop\evs\data\focus\csv"
CSV_NAME = "focus_csv.csv"
PICTURE_DIR = r"C:\Users\13119\OneDrive\Desktop\evs\data\focus\picture"

# =============================
# 1. 读取CSV事件数据
# =============================
def load_events(csv_path):
    df = pd.read_csv(csv_path)
    return df


# =============================
# 1b. EVS 采集相关
# =============================
def create_devices_with_tries():
    tries = 0
    tries_max = 6
    sleep_time_secs = 10
    while tries < tries_max:
        devices = system.create_device()
        if not devices:
            print(
                f"Try {tries + 1} of {tries_max}: waiting for {sleep_time_secs} secs for a device to be connected!"
            )
            for sec_count in range(sleep_time_secs):
                time.sleep(1)
                print(f"{sec_count + 1} seconds passed", "." * sec_count, end="\r")
            tries += 1
        else:
            print(f"Created {len(devices)} device(s)")
            return devices
    raise Exception("No device found! Please connect a device and run again.")


def configure_camera_for_xytp(device):
    nodemap = device.nodemap
    tl_stream_nodemap = device.tl_stream_nodemap

    nodemap["AcquisitionMode"].value = "Continuous"
    tl_stream_nodemap["StreamBufferHandlingMode"].value = "NewestOnly"
    nodemap["EventFormat"].value = "EVT3_0"
    nodemap["ErcEnable"].value = True
    nodemap["ErcRateLimit"].value = 10.0
    tl_stream_nodemap["StreamEvsOutputFormat"].value = "XYTPFrame"

    if ENABLE_BIAS_PRESET:
        apply_bias_preset(nodemap)


def _try_set_node_delta(nodemap, name, delta):
    try:
        node = nodemap.get_node(name)
    except Exception:
        return False
    try:
        cur = node.value
        if isinstance(cur, bool):
            return False
        new_val = cur + delta
        try:
            new_val = min(node.max, new_val)
        except Exception:
            pass
        try:
            new_val = max(node.min, new_val)
        except Exception:
            pass
        node.value = new_val
        print(f"[bias] {name}: {cur} -> {new_val}")
        return True
    except Exception as exc:
        print(f"[bias] {name} set failed: {exc}")
        return False


def apply_bias_preset(nodemap):
    applied = 0
    for name, delta in BIAS_DELTAS.items():
        if _try_set_node_delta(nodemap, name, delta):
            applied += 1
    if applied == 0:
        print("[bias] No bias nodes matched; preset not applied")


def parse_xytp_buffer(buffer, width, height):
    events = []
    channels_per_pixel = 4
    bytes_per_pixel = buffer.bits_per_pixel // 8
    valid_event_size = buffer.size_filled // bytes_per_pixel

    if valid_event_size == 0:
        return events

    src_data = ctypes.cast(buffer.pdata, ctypes.POINTER(ctypes.c_float))

    for i in range(valid_event_size):
        offset = i * channels_per_pixel
        x = src_data[offset + 0]
        y = src_data[offset + 1]
        t = src_data[offset + 2]
        p = src_data[offset + 3]

        x_int = int(round(x))
        y_int = int(round(y))
        p_int = 1 if p > 0.5 else 0

        if 0 <= x_int < width and 0 <= y_int < height:
            events.append([t, x_int, y_int, p_int])

    return events


def to_cdframe_uint8(diff):
    abs_diff = np.abs(diff)
    scale = float(np.percentile(abs_diff, 99)) if abs_diff.size > 0 else 1.0
    if scale <= 0:
        scale = 1.0
    img = 128.0 + (diff / scale) * 127.0
    img = np.clip(img, 0, 255)
    return img.astype(np.uint8)


def _gamma_u8(x_u8, gamma=0.75):
    x = x_u8.astype(np.float32) / 255.0
    x = np.clip(x, 0.0, 1.0) ** gamma
    return (x * 255.0).astype(np.uint8)


def _render_event_rgb(on_plane, off_plane, percentile=99, gamma=None):
    on_scale = float(np.percentile(on_plane, percentile)) if on_plane.max() > 0 else 1.0
    off_scale = float(np.percentile(off_plane, percentile)) if off_plane.max() > 0 else 1.0
    if on_scale <= 0:
        on_scale = 1.0
    if off_scale <= 0:
        off_scale = 1.0

    on_u8 = np.clip((on_plane / on_scale) * 255.0, 0, 255).astype(np.uint8)
    off_u8 = np.clip((off_plane / off_scale) * 255.0, 0, 255).astype(np.uint8)
    if gamma is not None:
        on_u8 = _gamma_u8(on_u8, gamma)
        off_u8 = _gamma_u8(off_u8, gamma)

    event_rgb = np.full((on_plane.shape[0], on_plane.shape[1], 3), 255, dtype=np.float32)
    on_alpha = (on_u8.astype(np.float32) / 255.0)[..., None]
    off_alpha = (off_u8.astype(np.float32) / 255.0)[..., None]
    red = np.array([0, 0, 255], dtype=np.float32)
    blue = np.array([255, 0, 0], dtype=np.float32)
    event_rgb = event_rgb * (1.0 - on_alpha) + red * on_alpha
    event_rgb = event_rgb * (1.0 - off_alpha) + blue * off_alpha
    return np.clip(event_rgb, 0, 255).astype(np.uint8)


def _denoise_event_planes(on_plane, off_plane):
    # Remove weak and isolated events to reduce speckle noise
    on = on_plane.copy()
    off = off_plane.copy()
    before_on = float(on.sum())
    before_off = float(off.sum())

    if on.max() > 0:
        on_min = max(1.0, float(np.percentile(on, 80)) * 0.1)
        on[on < on_min] = 0.0
    if off.max() > 0:
        off_min = max(1.0, float(np.percentile(off, 80)) * 0.1)
        off[off < off_min] = 0.0

    kernel = np.ones((3, 3), np.uint8)
    on_mask = (on > 0).astype(np.uint8)
    off_mask = (off > 0).astype(np.uint8)
    on_mask = cv2.morphologyEx(on_mask, cv2.MORPH_OPEN, kernel)
    off_mask = cv2.morphologyEx(off_mask, cv2.MORPH_OPEN, kernel)

    on *= on_mask
    off *= off_mask

    after_on = float(on.sum())
    after_off = float(off.sum())
    # If denoise removed almost everything, keep original planes
    if (before_on > 0 and after_on < before_on * 0.1) and (before_off > 0 and after_off < before_off * 0.1):
        return on_plane, off_plane

    return on, off


def set_diopter_raw(lens, diopter):
    if lens is None or not lens.is_connected():
        return
    if lens.mode != 5:
        lens.to_focal_power_mode()

    if lens.firmware_type == "A":
        raw_diopter = int((diopter + 5) * 200)
    else:
        raw_diopter = int(diopter * 200)

    lens.send_command(b"PwDA" + struct.pack(">h", raw_diopter) + b"\x00\x00")


# =============================
# ELP 滤波辅助函数（参考 EVK4_ELP/utils/filter.py）
# =============================
def _adaptive_filter(data, previous_value, threshold, window_average, alpha=0.5):
    if len(data) < 2:
        return data[-1]
    if abs(data[-1] - window_average) > threshold:
        return data[-1]
    return alpha * data[-1] + (1 - alpha) * previous_value


def _update_filter_elp(laplacian_product, filter_lp, threshold, alpha=0.5, window_size=10):
    if len(laplacian_product) == 1:
        filter_lp.append(laplacian_product[-1])
        return filter_lp
    if len(filter_lp) < window_size:
        window_avg = sum(laplacian_product) / len(laplacian_product)
    else:
        window_avg = sum(laplacian_product[-window_size:]) / window_size
    new_val = _adaptive_filter(laplacian_product, filter_lp[-1], threshold, window_avg, alpha)
    filter_lp.append(new_val)
    return filter_lp


def _diopter_at_time_ms(time_ms, start_diopter, end_diopter, duration_ms):
    if duration_ms <= 0:
        return start_diopter
    half_ms = duration_ms / 2.0
    if time_ms <= half_ms:
        ratio = max(0.0, min(time_ms / half_ms, 1.0))
        return start_diopter + (end_diopter - start_diopter) * ratio
    ratio = max(0.0, min((time_ms - half_ms) / half_ms, 1.0))
    return end_diopter + (start_diopter - end_diopter) * ratio


def _savgol_filter_1d(values, window_length, polyorder):
    if not values:
        return []
    if window_length < 3:
        return list(values)
    n = len(values)
    if n < window_length:
        return list(values)
    if window_length % 2 == 0:
        window_length += 1
    if window_length > n:
        window_length = n if n % 2 == 1 else n - 1
    if window_length < 3:
        return list(values)
    if polyorder >= window_length:
        polyorder = window_length - 1

    half = window_length // 2
    x = np.arange(-half, half + 1, dtype=np.float64)
    a = np.vander(x, polyorder + 1, increasing=True)
    coeffs = np.linalg.pinv(a)[0]
    padded = np.pad(np.asarray(values, dtype=np.float64), (half, half), mode="edge")
    smooth = np.convolve(padded, coeffs[::-1], mode="valid")
    return smooth.tolist()


def _find_peaks_with_constraints(scores, event_counts, min_score, min_events, trend_window):
    peaks = []
    n = len(scores)
    if n < 3:
        return peaks
    for i in range(1, n - 1):
        if scores[i] <= scores[i - 1] or scores[i] <= scores[i + 1]:
            continue
        if scores[i] < min_score:
            continue
        if event_counts and i < len(event_counts) and event_counts[i] < min_events:
            continue
        if trend_window > 1:
            left_start = i - trend_window
            right_end = i + trend_window
            if left_start < 0 or right_end >= n:
                continue
            left = scores[left_start:i + 1]
            right = scores[i:right_end + 1]
            left_slope = np.polyfit(np.arange(len(left)), left, 1)[0]
            right_slope = np.polyfit(np.arange(len(right)), right, 1)[0]
            if left_slope <= 0 or right_slope >= 0:
                continue
        peaks.append(i)
    return peaks


def _estimate_focus_time_ms(time_ms_list, scores, frame_interval_ms, best_idx=None):
    if not scores:
        return None, None
    if best_idx is None:
        best_idx = int(np.argmax(scores))
    raw_ms = time_ms_list[best_idx]

    # Sub-frame refinement using quadratic interpolation around the peak
    refined_ms = raw_ms
    if 0 < best_idx < len(scores) - 1:
        s0 = scores[best_idx - 1]
        s1 = scores[best_idx]
        s2 = scores[best_idx + 1]
        denom = (s0 - 2 * s1 + s2)
        if denom != 0:
            delta = 0.5 * (s0 - s2) / denom
            delta = max(min(delta, 0.5), -0.5)
            refined_ms = raw_ms + delta * frame_interval_ms

    # Align to window center and optional lens lag compensation
    aligned_ms = refined_ms - frame_interval_ms / 2.0 + FOCUS_ALIGN_LAG_MS
    return raw_ms, aligned_ms


def _detect_focus_index(time_ms_list, scores, filtered_scores):
    if not scores:
        return None

    if FOCUS_DETECT_MODE == "zero_cross" and filtered_scores is not None:
        arr = np.array(filtered_scores, dtype=np.float32)
        if len(arr) > 5:
            tail_min = float(np.min(arr[-5:]))
            head_max = float(np.max(arr[:-5]))
            if tail_min < ZERO_CROSS_NEG_THRESHOLD and head_max > ZERO_CROSS_POS_THRESHOLD:
                transitions = np.where((arr[:-1] > 0) & (arr[1:] < 0))[0] + 1
                if transitions.size > 0:
                    return int(transitions[-1])

    return int(np.argmax(scores))


def _collect_transitions(time_ms_list, scores, filtered_scores, frame_interval_ms, duration_ms):
    if not scores:
        return []

    arr = np.array(filtered_scores if filtered_scores else scores, dtype=np.float32)
    transitions = []
    seen = set()
    for i in range(1, len(arr)):
        prev_val = float(arr[i - 1])
        curr_val = float(arr[i])
        if prev_val > 0 and curr_val < 0:
            if TRANSITION_USE_THRESH:
                if not (prev_val > TRANSITION_MIN_POS or curr_val < TRANSITION_MAX_NEG):
                    continue
            raw_ms = time_ms_list[i]
            if raw_ms < MIN_SEARCH_MS:
                continue
            aligned_ms = raw_ms - frame_interval_ms / 2.0 + FOCUS_ALIGN_LAG_MS
            aligned_ms = max(0.0, min(aligned_ms, duration_ms))
            ratio = max(0.0, min(aligned_ms / duration_ms, 1.0))
            transitions.append((i, raw_ms, aligned_ms, ratio, "zero_cross"))
            seen.add(i)

    if RECORD_JUMPS:
        for i in range(1, len(arr)):
            if i in seen:
                continue
            prev_val = float(arr[i - 1])
            curr_val = float(arr[i])
            if abs(curr_val - prev_val) >= TRANSITION_JUMP_THRESHOLD:
                raw_ms = time_ms_list[i]
                if raw_ms < MIN_SEARCH_MS:
                    continue
                aligned_ms = raw_ms - frame_interval_ms / 2.0 + FOCUS_ALIGN_LAG_MS
                aligned_ms = max(0.0, min(aligned_ms, duration_ms))
                ratio = max(0.0, min(aligned_ms / duration_ms, 1.0))
                transitions.append((i, raw_ms, aligned_ms, ratio, "jump"))

    return transitions


def _sharpness_metric(diff):
    lap = cv2.Laplacian(diff.astype(np.float32), cv2.CV_32F, ksize=3)
    return float(np.var(lap))


def _gradient_energy(diff):
    gx = cv2.Sobel(diff.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(diff.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    return float(np.mean(gx * gx + gy * gy))


def _accumulate_window(all_events, t_start, t_end, height, width, use_roi, roi):
    on = np.zeros((height, width), dtype=np.float32)
    off = np.zeros((height, width), dtype=np.float32)
    for t, x, y, p in all_events:
        if t < t_start:
            continue
        if t > t_end:
            break
        if use_roi:
            y1, y2, x1, x2 = roi
            if not (y1 <= y < y2 and x1 <= x < x2):
                continue
        if p == 1:
            on[y, x] += 1.0
        else:
            off[y, x] += 1.0
    return on, off


def _accumulate_with_min_events(all_events, center_t, base_window_ms, height, width, use_roi, roi,
                                min_events, max_mult):
    best_on = None
    best_off = None
    best_mult = 1
    for mult in range(1, max_mult + 1):
        window_ms = base_window_ms * mult
        t_start = center_t - window_ms * 1000.0
        t_end = center_t + window_ms * 1000.0
        on, off = _accumulate_window(all_events, t_start, t_end, height, width, use_roi, roi)
        ev_count = float(on.sum() + off.sum())
        best_on, best_off, best_mult = on, off, mult
        if ev_count >= min_events:
            break
    return best_on, best_off, best_mult


def _select_best_transition_by_sharpness(all_events, t0, transitions, height, width, window_ms, use_roi, roi):
    best_idx = None
    best_raw_ms = None
    best_aligned_ms = None
    best_metric = -1.0
    for idx, raw_ms, aligned_ms, _, _ in transitions:
        center_t = t0 + aligned_ms * 1000.0
        t_start = center_t - window_ms * 1000.0
        t_end = center_t + window_ms * 1000.0
        on, off = _accumulate_window(all_events, t_start, t_end, height, width, use_roi, roi)
        metric = _sharpness_metric(on - off)
        if metric > best_metric:
            best_metric = metric
            best_idx = idx
            best_raw_ms = raw_ms
            best_aligned_ms = aligned_ms
    return best_idx, best_raw_ms, best_aligned_ms, best_metric


def _estimate_lag_by_sharpness(all_events, t0, base_time_ms, height, width,
                               window_ms, search_range_ms, step_ms):
    if not all_events:
        return None, None

    base_center = t0 + base_time_ms * 1000.0
    min_t = base_center - (search_range_ms + window_ms) * 1000.0
    max_t = base_center + (search_range_ms + window_ms) * 1000.0

    window_events = []
    for e in all_events:
        if e[0] < min_t:
            continue
        if e[0] > max_t:
            break
        window_events.append(e)
    if not window_events:
        return None, None

    best_lag = 0.0
    best_metric = -1.0
    for lag_ms in range(-search_range_ms, search_range_ms + 1, step_ms):
        center_t = base_center + lag_ms * 1000.0
        t_start = center_t - window_ms * 1000.0
        t_end = center_t + window_ms * 1000.0

        on = np.zeros((height, width), dtype=np.float32)
        off = np.zeros((height, width), dtype=np.float32)
        for t, x, y, p in window_events:
            if t < t_start:
                continue
            if t > t_end:
                break
            if p == 1:
                on[y, x] += 1.0
            else:
                off[y, x] += 1.0

        metric = _sharpness_metric(on - off)
        if metric > best_metric:
            best_metric = metric
            best_lag = float(lag_ms)

    return best_lag, best_metric


def _compute_scores_from_events(all_events, width, height, duration_ms, frame_interval_ms):
    if not all_events:
        return (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            np.zeros((height, width), dtype=np.float32),
            np.zeros((height, width), dtype=np.float32),
        )

    all_events.sort(key=lambda e: e[0])
    t0 = all_events[0][0]
    next_frame_ms = frame_interval_ms
    idx = 0
    total_on = np.zeros((height, width), dtype=np.float32)
    total_off = np.zeros((height, width), dtype=np.float32)
    accumulated_on = np.zeros((height, width), dtype=np.float32)
    accumulated_off = np.zeros((height, width), dtype=np.float32)

    laplacian_product = []
    time_ms_list = []
    frame_snaps = []
    filter_laplacian_product = []
    grad_energy_list = []
    corr_score_list = []
    event_count_list = []

    while next_frame_ms <= duration_ms:
        t_end = t0 + next_frame_ms * 1000.0
        while idx < len(all_events) and all_events[idx][0] < t_end:
            _, x, y, p = all_events[idx]
            idx += 1
            if not (0 <= x < width and 0 <= y < height):
                continue
            if USE_ROI:
                y1, y2, x1, x2 = ROI
                if not (y1 <= y < y2 and x1 <= x < x2):
                    continue
            if p == 1:
                accumulated_on[y, x] += 1.0
                total_on[y, x] += 1.0
            else:
                accumulated_off[y, x] += 1.0
                total_off[y, x] += 1.0

        diff = accumulated_on - accumulated_off
        n_ev = float(accumulated_on.sum() + accumulated_off.sum())
        if n_ev < 200:
            score = 0.0
            grad_energy = 0.0
            corr_score = 0.0
        else:
            blurred = cv2.GaussianBlur(diff, (9, 9), 2.0)
            laplacian = cv2.Laplacian(blurred, cv2.CV_32F, ksize=3)
            score = float(-np.sum(laplacian * blurred))
            grad_energy = _gradient_energy(blurred)
            if USE_ROI:
                y1, y2, x1, x2 = ROI
                if y2 > y1 and x2 > x1:
                    roi_lap = laplacian[y1:y2, x1:x2]
                    roi_diff = diff[y1:y2, x1:x2]
                else:
                    roi_lap = laplacian
                    roi_diff = diff
            else:
                roi_lap = laplacian
                roi_diff = diff
            corr_score = float(np.sum(np.abs(roi_lap) * np.abs(roi_diff)))

        laplacian_product.append(score)
        if FILTER_ELP:
            filter_laplacian_product = _update_filter_elp(
                laplacian_product,
                filter_laplacian_product,
                FILTER_THRESHOLD,
                alpha=FILTER_ALPHA,
                window_size=FILTER_WINDOW,
            )
        grad_energy_list.append(grad_energy)
        corr_score_list.append(corr_score)
        event_count_list.append(n_ev)
        time_ms_list.append(next_frame_ms)
        frame_snaps.append((accumulated_on.copy(), accumulated_off.copy()))

        accumulated_on = np.zeros((height, width), dtype=np.float32)
        accumulated_off = np.zeros((height, width), dtype=np.float32)
        next_frame_ms += frame_interval_ms

    return (
        laplacian_product,
        time_ms_list,
        frame_snaps,
        filter_laplacian_product,
        grad_energy_list,
        corr_score_list,
        event_count_list,
        total_on,
        total_off,
    )


def _finalize_focus_results(
    all_events,
    width,
    height,
    duration_ms,
    frame_interval_ms,
    start_diopter,
    end_diopter,
    picture_dir,
    laplacian_product,
    time_ms_list,
    frame_snaps,
    filter_laplacian_product,
    grad_energy_list,
    corr_score_list,
    event_count_list,
    total_on,
    total_off,
):
    transitions = []
    best_idx = None
    focus_time_ms = None
    aligned_focus_time_ms = None
    print(f"\n[DEBUG] 总帧数: {len(laplacian_product)}")
    corr_smoothed = []
    corr_peaks = []
    corr_best_idx = None
    if laplacian_product:
        print(f"[DEBUG] ELP 范围: min={min(laplacian_product):+.4f}  max={max(laplacian_product):+.4f}")
        corr_smoothed = _savgol_filter_1d(corr_score_list, CORR_SG_WINDOW, CORR_SG_POLYORDER)
        if corr_smoothed:
            peak_min_score = max(CORR_PEAK_MIN_SCORE, max(corr_smoothed) * CORR_PEAK_MIN_RATIO)
            corr_peaks = _find_peaks_with_constraints(
                corr_smoothed,
                event_count_list,
                peak_min_score,
                CORR_PEAK_MIN_EVENTS,
                CORR_TREND_WINDOW,
            )
            if corr_peaks:
                corr_best_idx = max(corr_peaks, key=lambda i: corr_smoothed[i])
            else:
                corr_best_idx = int(np.argmax(corr_smoothed))
        curve_csv = os.path.join(picture_dir, "focus_elp_curve.csv")
        with open(curve_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["time_ms", "elp_score", "elp_filtered", "grad_energy", "corr_score", "corr_smooth"])
            for i, t_ms in enumerate(time_ms_list):
                filt = filter_laplacian_product[i] if filter_laplacian_product else ""
                grad = grad_energy_list[i] if i < len(grad_energy_list) else ""
                corr = corr_score_list[i] if i < len(corr_score_list) else ""
                corr_smooth = corr_smoothed[i] if i < len(corr_smoothed) else ""
                writer.writerow([t_ms, laplacian_product[i], filt, grad, corr, corr_smooth])
        print(f"Saved ELP curve CSV: {curve_csv}")

        plt.figure(figsize=(8, 4))
        plt.plot(time_ms_list, laplacian_product, marker="o", markersize=3, linewidth=1, label="ELP raw")
        if filter_laplacian_product:
            plt.plot(time_ms_list, filter_laplacian_product, marker=".", markersize=2, linewidth=1, label="ELP filtered")
        plt.xlabel("time (ms)")
        plt.ylabel("ELP score")
        plt.title("ELP curve")
        plt.grid(True)
        if filter_laplacian_product:
            plt.legend(loc="best")
        elp_png = os.path.join(picture_dir, "focus_elp_curve.png")
        plt.tight_layout()
        plt.savefig(elp_png, dpi=150)
        plt.close()
        print(f"Saved ELP curve image: {elp_png}")

        fig, ax1 = plt.subplots(figsize=(8, 4))
        ax1.plot(time_ms_list, laplacian_product, marker="o", markersize=3, linewidth=1, label="ELP raw")
        if filter_laplacian_product:
            ax1.plot(time_ms_list, filter_laplacian_product, marker=".", markersize=2, linewidth=1, label="ELP filtered")
        ax1.set_xlabel("time (ms)")
        ax1.set_ylabel("ELP score")
        ax1.set_title("ELP vs gradient energy")
        ax1.grid(True)

        ax2 = ax1.twinx()
        if grad_energy_list:
            ax2.plot(time_ms_list, grad_energy_list, marker="x", markersize=3, linewidth=1,
                     color="tab:green", label="Grad energy")
        ax2.set_ylabel("Gradient energy")

        lines = ax1.get_lines() + ax2.get_lines()
        labels = [line.get_label() for line in lines]
        if lines:
            ax1.legend(lines, labels, loc="best")
        compare_png = os.path.join(picture_dir, "focus_elp_vs_grad.png")
        fig.tight_layout()
        fig.savefig(compare_png, dpi=150)
        plt.close(fig)
        print(f"Saved compare curve image: {compare_png}")

        if corr_score_list:
            plt.figure(figsize=(8, 4))
            plt.plot(time_ms_list, corr_score_list, marker="o", markersize=3, linewidth=1, label="corr raw")
            if corr_smoothed:
                plt.plot(time_ms_list, corr_smoothed, marker=".", markersize=2, linewidth=1, label="corr smooth")
            if corr_peaks:
                peak_x = [time_ms_list[i] for i in corr_peaks]
                peak_y = [corr_smoothed[i] if corr_smoothed else corr_score_list[i] for i in corr_peaks]
                plt.scatter(peak_x, peak_y, s=30, color="tab:red", label="peaks")
            plt.xlabel("time (ms)")
            plt.ylabel("Corr score")
            plt.title("Event-gradient correlation")
            plt.grid(True)
            plt.legend(loc="best")
            corr_png = os.path.join(picture_dir, "focus_corr_curve.png")
            plt.tight_layout()
            plt.savefig(corr_png, dpi=150)
            plt.close()
            print(f"Saved corr curve image: {corr_png}")

        if RECORD_ALL_TRANSITIONS:
            transitions = _collect_transitions(
                time_ms_list,
                laplacian_product,
                filter_laplacian_product if FILTER_ELP else None,
                frame_interval_ms,
                duration_ms,
            )
            if not transitions:
                print("No transitions found for image export")
            trans_csv = os.path.join(picture_dir, "focus_transitions.csv")
            with open(trans_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["index", "time_ms_raw", "time_ms_aligned", "diopter", "type"])
                for idx, raw_ms, aligned_ms, ratio, kind in transitions:
                    diopter = _diopter_at_time_ms(aligned_ms, start_diopter, end_diopter, duration_ms)
                    writer.writerow([idx, f"{raw_ms:.1f}", f"{aligned_ms:.1f}", f"{diopter:.3f}", kind])
            print(f"Saved transitions CSV: {trans_csv} (count={len(transitions)})")

        if transitions and all_events and SELECT_TRANSITION_BY_SHARPNESS:
            all_events.sort(key=lambda e: e[0])
            t0 = all_events[0][0]
            best_idx, best_raw_ms, best_aligned_ms, best_metric = _select_best_transition_by_sharpness(
                all_events,
                t0,
                transitions,
                height,
                width,
                FOCUS_WINDOW_MS,
                USE_ROI,
                ROI,
            )
            if best_idx is not None:
                focus_time_ms = best_raw_ms
                aligned_focus_time_ms = best_aligned_ms
                print(f"Selected transition: idx={best_idx} sharpness={best_metric:.2f}")

            if TRANSITION_SAVE_IMAGES and transitions and all_events:
                trans_dir = os.path.join(picture_dir, TRANSITION_IMAGE_DIR)
                os.makedirs(trans_dir, exist_ok=True)
                all_events.sort(key=lambda e: e[0])
                t0 = all_events[0][0]

                for idx, raw_ms, aligned_ms, _, kind in transitions:
                    center_t = t0 + aligned_ms * 1000.0
                    t_start = center_t - TRANSITION_WINDOW_MS * 1000.0
                    t_end = center_t + TRANSITION_WINDOW_MS * 1000.0
                    on, off = _accumulate_window(
                        all_events, t_start, t_end, height, width, USE_ROI, ROI
                    )

                    event_rgb = _render_event_rgb(on, off, percentile=99, gamma=None)
                    event_path = os.path.join(trans_dir, f"transition_{idx:03d}_{kind}_event.png")
                    cv2.imwrite(event_path, event_rgb)
                print(f"Saved transition images to: {trans_dir}")
        if corr_best_idx is not None:
            best_idx = corr_best_idx
            focus_time_ms, aligned_focus_time_ms = _estimate_focus_time_ms(
                time_ms_list, corr_smoothed if corr_smoothed else corr_score_list, frame_interval_ms, best_idx=best_idx
            )
            best_corr = corr_smoothed[best_idx] if corr_smoothed else corr_score_list[best_idx]
            print(f"[focus] Corr peak idx={best_idx} score={best_corr:.2f}")

    focus_on = focus_off = None
    if laplacian_product and best_idx is None:
        best_idx = _detect_focus_index(
            time_ms_list, laplacian_product, filter_laplacian_product if FILTER_ELP else None
        )
        focus_time_ms, aligned_focus_time_ms = _estimate_focus_time_ms(
            time_ms_list, laplacian_product, frame_interval_ms, best_idx=best_idx
        )
    if laplacian_product and best_idx is not None:
        focus_on, focus_off = frame_snaps[best_idx]

    if focus_on is not None and all_events and aligned_focus_time_ms is not None:
        diopter_at_focus = _diopter_at_time_ms(aligned_focus_time_ms, start_diopter, end_diopter, duration_ms)
        best_score = laplacian_product[best_idx]
        best_corr = None
        if corr_smoothed:
            best_corr = corr_smoothed[best_idx]
        elif corr_score_list:
            best_corr = corr_score_list[best_idx]
        print(f"\n===== 对焦结果 =====")
        print(f"Focus moment(raw): {focus_time_ms:.1f} ms")
        print(f"Focus moment     : {aligned_focus_time_ms:.1f} ms  (diopter ≈ {diopter_at_focus:.2f})")
        if best_corr is not None:
            print(f"Best corr score : {best_corr:.4f}")
        print(f"Best ELP score  : {best_score:.4f}")

        all_events.sort(key=lambda e: e[0])
        t0 = all_events[0][0]

        suggested_lag, lag_metric = _estimate_lag_by_sharpness(
            all_events,
            t0,
            aligned_focus_time_ms,
            height,
            width,
            FOCUS_WINDOW_MS,
            ALIGN_SEARCH_RANGE_MS,
            ALIGN_SEARCH_STEP_MS,
        )
        if suggested_lag is not None:
            print(f"Suggested lag    : {suggested_lag:+.1f} ms (sharpness={lag_metric:.2f})")
            if APPLY_ESTIMATED_LAG:
                aligned_focus_time_ms += suggested_lag
                print(f"Applied lag      : {suggested_lag:+.1f} ms")

        center_t = t0 + aligned_focus_time_ms * 1000.0
        t_start = center_t - FOCUS_WINDOW_MS * 1000.0
        t_end = center_t + FOCUS_WINDOW_MS * 1000.0

        if MIN_FOCUS_EVENTS > 0 and MAX_FOCUS_WINDOW_MULT > 1:
            focus_win_on, focus_win_off, used_mult = _accumulate_with_min_events(
                all_events,
                center_t,
                FOCUS_WINDOW_MS,
                height,
                width,
                USE_ROI,
                ROI,
                MIN_FOCUS_EVENTS,
                MAX_FOCUS_WINDOW_MULT,
            )
            if used_mult > 1:
                print(f"[focus] Adaptive window: x{used_mult} (min_events={MIN_FOCUS_EVENTS})")
        else:
            focus_win_on, focus_win_off = _accumulate_window(
                all_events, t_start, t_end, height, width, USE_ROI, ROI
            )

        focus_window_events = int(focus_win_on.sum() + focus_win_off.sum())
        print(f"[focus] Window events: {focus_window_events}")

        if focus_window_events == 0 and frame_snaps and event_count_list:
            fallback_idx = int(np.argmax(event_count_list))
            fallback_events = float(event_count_list[fallback_idx])
            if fallback_events > 0:
                focus_win_on, focus_win_off = frame_snaps[fallback_idx]
                focus_window_events = int(focus_win_on.sum() + focus_win_off.sum())
                print(
                    f"[focus] Window empty; fallback to max-event frame idx={fallback_idx} "
                    f"events={int(fallback_events)}"
                )

        if SAVE_FOCUS_GRAY:
            focus_diff = focus_win_on - focus_win_off
            gray = to_cdframe_uint8(focus_diff)
            gray_path = os.path.join(picture_dir, "focus_pic_gray.png")
            cv2.imwrite(gray_path, gray)
            print(f"Saved grayscale image : {gray_path}")

        render_on, render_off = focus_win_on, focus_win_off
        if EVENT_RENDER_DENOISE:
            render_on, render_off = _denoise_event_planes(render_on, render_off)
        event_rgb = _render_event_rgb(
            render_on,
            render_off,
            percentile=EVENT_RENDER_PERCENTILE,
            gamma=EVENT_RENDER_GAMMA,
        )
        event_path = os.path.join(picture_dir, "focus_pic_event.png")
        cv2.imwrite(event_path, event_rgb)
        print(f"Saved event frame image: {event_path}")
    else:
        print("No frames processed, cannot save focus images")

    render_total_on, render_total_off = total_on, total_off
    if EVENT_RENDER_DENOISE:
        render_total_on, render_total_off = _denoise_event_planes(
            render_total_on, render_total_off
        )
    total_rgb = _render_event_rgb(
        render_total_on,
        render_total_off,
        percentile=EVENT_RENDER_PERCENTILE,
        gamma=EVENT_RENDER_GAMMA,
    )
    total_path = os.path.join(picture_dir, "focus_total_event.png")
    cv2.imwrite(total_path, total_rgb)
    print(f"Saved total event image: {total_path}")


def collect_focus_data(
    duration_ms=CAPTURE_DURATION_MS,
    frame_interval_ms=FRAME_INTERVAL_MS,
    diopter_update_ms=DIOPTER_UPDATE_MS,
    start_diopter=START_DIOPTER,
    end_diopter=END_DIOPTER,
    csv_dir=CSV_DIR,
    csv_name=CSV_NAME,
    picture_dir=PICTURE_DIR,
):
    lens = Lens("COM3", debug=False)
    if lens.is_connected() and lens.camera_device is not None:
        device = lens.camera_device
    else:
        devices = create_devices_with_tries()
        device = system.select_device(devices)

    nodemap = device.nodemap
    width = nodemap["Width"].value
    height = nodemap["Height"].value

    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(picture_dir, exist_ok=True)

    configure_camera_for_xytp(device)

    print(f"Camera resolution: {width} x {height}")
    half_ms = duration_ms / 2.0
    print(
        f"Diopter ramp (two-phase): {start_diopter} -> {end_diopter} over {half_ms:.0f}ms, "
        f"then {end_diopter} -> {start_diopter} over {half_ms:.0f}ms"
    )
    print(f"Saving CSV to: {os.path.join(csv_dir, csv_name)}")
    print(f"Saving frames to: {picture_dir}")

    device.start_stream(50)
    stream_started = True

    start_time = time.time()

    # 先开始记录，再设初始屈光度
    if lens is not None and lens.is_connected():
        if lens.mode != 5:
            lens.to_focal_power_mode()
        lens.set_diopter(start_diopter)
        print(f"[init] 初始屈光度设为 {start_diopter}")

    next_frame_ms = frame_interval_ms
    frame_count = 0
    all_events = []
    timeout_count = 0

    accumulated_on = np.zeros((height, width), dtype=np.float32)
    accumulated_off = np.zeros((height, width), dtype=np.float32)
    total_on = np.zeros((height, width), dtype=np.float32)
    total_off = np.zeros((height, width), dtype=np.float32)

    # ELP 分数列表、时间戳列表、帧快照
    laplacian_product = []
    time_ms_list = []
    frame_snaps = []  # 每帧的 (on, off) 快照
    filter_laplacian_product = []
    grad_energy_list = []
    corr_score_list = []
    event_count_list = []

    # 镜头控制线程：线性变化屈光度
    stop_lens_thread = False

    def lens_thread_fn():
        last_diopter = None
        half_ms_local = duration_ms / 2.0
        while not stop_lens_thread:
            elapsed_ms = (time.time() - start_time) * 1000
            if elapsed_ms >= duration_ms:
                break
            if elapsed_ms <= half_ms_local:
                ratio = max(0.0, min(elapsed_ms / half_ms_local, 1.0))
                target_diopter = start_diopter + (end_diopter - start_diopter) * ratio
            else:
                ratio = max(0.0, min((elapsed_ms - half_ms_local) / half_ms_local, 1.0))
                target_diopter = end_diopter + (start_diopter - end_diopter) * ratio
            if last_diopter is None or abs(target_diopter - last_diopter) >= 1e-3:
                try:
                    if lens is not None and lens.is_connected():
                        lens.set_diopter(target_diopter)
                    else:
                        set_diopter_raw(lens, target_diopter)
                except Exception as e:
                    print(f"  [lens] 设置屈光度失败: {e}")
                last_diopter = target_diopter
            time.sleep(max(diopter_update_ms, 1) / 1000.0)

    import threading
    lens_th = threading.Thread(target=lens_thread_fn, daemon=True)
    lens_th.start()

    try:
        while True:
            elapsed_ms = (time.time() - start_time) * 1000
            if elapsed_ms >= duration_ms:
                break

            try:
                buffer = device.get_buffer(timeout=5)
            except Exception as exc:
                timeout_count += 1
                if timeout_count <= 3:
                    print(f"  [stream] get_buffer timeout: {exc}")
                time.sleep(0.005)
                continue
            if buffer.is_incomplete:
                device.requeue_buffer(buffer)
                continue

            events = parse_xytp_buffer(buffer, width, height)
            if events:
                all_events.extend(events)
                for _, x, y, p in events:
                    if USE_ROI:
                        y1, y2, x1, x2 = ROI
                        if not (y1 <= y < y2 and x1 <= x < x2):
                            continue
                    if p == 1:
                        accumulated_on[y, x] += 1.0
                        total_on[y, x] += 1.0
                    else:
                        accumulated_off[y, x] += 1.0
                        total_off[y, x] += 1.0

            elapsed_ms = (time.time() - start_time) * 1000
            while elapsed_ms >= next_frame_ms and next_frame_ms < duration_ms:
                diff = accumulated_on - accumulated_off
                n_ev = float(accumulated_on.sum() + accumulated_off.sum())
                if n_ev < 200:
                    # 事件数太少（跳变前的噪声期）直接计为0，不参与对焦检测
                    score = 0.0
                    grad_energy = 0.0
                    corr_score = 0.0
                else:
                    blurred = cv2.GaussianBlur(diff, (9, 9), 2.0)
                    laplacian = cv2.Laplacian(blurred, cv2.CV_32F, ksize=3)
                    # 不除以事件数：归一化会使跳变后密集帧分数反而下降
                    score = float(-np.sum(laplacian * blurred))
                    grad_energy = _gradient_energy(blurred)
                    if USE_ROI:
                        y1, y2, x1, x2 = ROI
                        if y2 > y1 and x2 > x1:
                            roi_lap = laplacian[y1:y2, x1:x2]
                            roi_diff = diff[y1:y2, x1:x2]
                        else:
                            roi_lap = laplacian
                            roi_diff = diff
                    else:
                        roi_lap = laplacian
                        roi_diff = diff
                    corr_score = float(np.sum(np.abs(roi_lap) * np.abs(roi_diff)))
                laplacian_product.append(score)
                if FILTER_ELP:
                    filter_laplacian_product = _update_filter_elp(
                        laplacian_product,
                        filter_laplacian_product,
                        FILTER_THRESHOLD,
                        alpha=FILTER_ALPHA,
                        window_size=FILTER_WINDOW,
                    )
                grad_energy_list.append(grad_energy)
                corr_score_list.append(corr_score)
                event_count_list.append(n_ev)
                time_ms_list.append(next_frame_ms)
                frame_snaps.append((accumulated_on.copy(), accumulated_off.copy()))
                print(f"  frame {frame_count:03d} | t={next_frame_ms:6.1f}ms | "
                      f"raw={score:+.1f} | grad={grad_energy:.2f} | corr={corr_score:.1f} | "
                      f"on={int(accumulated_on.sum())} off={int(accumulated_off.sum())}")
                frame_count += 1
                accumulated_on = np.zeros((height, width), dtype=np.float32)
                accumulated_off = np.zeros((height, width), dtype=np.float32)
                next_frame_ms += frame_interval_ms

            device.requeue_buffer(buffer)
    finally:
        stop_lens_thread = True
        lens_th.join(timeout=1)
        if 'stream_started' in locals() and stream_started:
            try:
                device.stop_stream()
            except Exception as exc:
                print(f"Stop stream failed: {exc}")

        if lens is not None and lens.is_connected() and lens.camera_device is not None:
            try:
                lens.close_connection()
            except Exception as exc:
                print(f"Close lens connection failed: {exc}")
        else:
            try:
                system.destroy_device()
            except Exception as exc:
                print(f"Destroy device failed: {exc}")

    if all_events:
        all_events.sort(key=lambda e: e[0])
        csv_path = os.path.join(csv_dir, csv_name)
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["t", "x", "y", "p"])
            writer.writerows(all_events)
        print(f"Saved {len(all_events):,} events to {csv_path}")
    else:
        print("No events captured")

    _finalize_focus_results(
        all_events,
        width,
        height,
        duration_ms,
        frame_interval_ms,
        start_diopter,
        end_diopter,
        picture_dir,
        laplacian_product,
        time_ms_list,
        frame_snaps,
        filter_laplacian_product,
        grad_energy_list,
        corr_score_list,
        event_count_list,
        total_on,
        total_off,
    )


def analyze_csv_focus(
    csv_path,
    duration_ms=None,
    frame_interval_ms=FRAME_INTERVAL_MS,
    start_diopter=START_DIOPTER,
    end_diopter=END_DIOPTER,
    picture_dir=PICTURE_DIR,
):
    if not os.path.exists(csv_path):
        print(f"CSV not found: {csv_path}")
        return

    events_df = load_events(csv_path)
    if events_df is None or events_df.empty:
        print("CSV has no events")
        return

    all_events = []
    for t, x, y, p in events_df[["t", "x", "y", "p"]].values:
        all_events.append([float(t), int(x), int(y), int(p)])

    all_events.sort(key=lambda e: e[0])
    t_start_us = float(all_events[0][0])
    t_end_us = float(all_events[-1][0])
    csv_span_ms = max(0.0, (t_end_us - t_start_us) / 1000.0)
    if duration_ms is None or duration_ms <= 0:
        duration_ms = csv_span_ms
    if duration_ms <= 0:
        duration_ms = float(frame_interval_ms)

    print(
        f"[analyze_csv] CSV time range: {t_start_us:.0f} us -> {t_end_us:.0f} us "
        f"(span {csv_span_ms:.1f} ms)"
    )
    print(f"[analyze_csv] Analysis duration: {duration_ms:.1f} ms")

    os.makedirs(picture_dir, exist_ok=True)

    width = IMG_W
    height = IMG_H
    (
        laplacian_product,
        time_ms_list,
        frame_snaps,
        filter_laplacian_product,
        grad_energy_list,
        corr_score_list,
        event_count_list,
        total_on,
        total_off,
    ) = _compute_scores_from_events(all_events, width, height, duration_ms, frame_interval_ms)

    _finalize_focus_results(
        all_events,
        width,
        height,
        duration_ms,
        frame_interval_ms,
        start_diopter,
        end_diopter,
        picture_dir,
        laplacian_product,
        time_ms_list,
        frame_snaps,
        filter_laplacian_product,
        grad_energy_list,
        corr_score_list,
        event_count_list,
        total_on,
        total_off,
    )

# =============================
# 2. 事件 → 帧
# =============================
def events_to_frame(events, t_start, t_end):
    frame = np.zeros((IMG_H, IMG_W), dtype=np.float32)

    subset = events[(events['t'] >= t_start) & (events['t'] < t_end)]

    for _, e in subset.iterrows():
        x = int(e['x'])
        y = int(e['y'])
        p = int(e['p'])

        if 0 <= x < IMG_W and 0 <= y < IMG_H:
            frame[y, x] += p

    return frame

# =============================
# 3. ELP (Event Laplacian Product)
# =============================
def elp_score(event_frame):
    lap = cv2.Laplacian(event_frame.astype(np.float32), cv2.CV_32F, ksize=3)
    return float(-np.sum(lap * event_frame))


def elp_state(score):
    if score > 0:
        return "未对焦"
    if score < 0:
        return "已过焦"
    return "中性"

# =============================
# 4. 处理单个CSV（一个焦距）
# =============================
def process_focus(csv_path):
    events = load_events(csv_path)

    t_min = events['t'].min()
    t_max = events['t'].max()

    scores = []
    frames = []
    time_stamps = []

    t = t_min
    while t < t_max:
        frame = events_to_frame(events, t, t + TIME_WINDOW)
        score = elp_score(frame)

        scores.append(score)
        frames.append(frame)
        time_stamps.append(t)

        t += TIME_WINDOW

    # 返回平均得分 + 最清晰那一帧
    best_idx = np.argmax(scores)
    best_frame = frames[best_idx]

    return np.mean(scores), best_frame, scores, time_stamps

# =============================
# 5. 找最佳对焦
# =============================
def find_best_focus(folder_path):
    results = {}
    best_frames = {}
    focus_times = {}
    focus_states = {}

    for file in os.listdir(folder_path):
        if file.endswith(".csv"):
            focus_value = float(file.replace(".csv", ""))
            csv_path = os.path.join(folder_path, file)

            score, best_frame, scores, time_stamps = process_focus(csv_path)

            results[focus_value] = score
            best_frames[focus_value] = best_frame

            focus_time = None
            for idx in range(1, len(scores)):
                prev_score = scores[idx - 1]
                curr_score = scores[idx]
                if prev_score == 0:
                    continue
                if (prev_score > 0 and curr_score < 0) or (prev_score < 0 and curr_score > 0):
                    focus_time = time_stamps[idx]
                    break
            focus_times[focus_value] = focus_time

            last_non_zero = 0.0
            for score_value in reversed(scores):
                if score_value != 0:
                    last_non_zero = score_value
                    break
            focus_states[focus_value] = elp_state(last_non_zero)

            print(f"Focus: {focus_value:.3f}, Score: {score:.2f}")

    best_focus = max(results, key=results.get)
    worst_focus = min(results, key=results.get)

    print("\n===== 对焦结果 =====")
    print(f"Best Focus: {best_focus}")
    print(f"Worst Focus: {worst_focus}")

    return best_focus, worst_focus, results, best_frames, focus_times, focus_states

# =============================
# 6. 画清晰度曲线
# =============================
def plot_focus_curve(results):
    x = sorted(results.keys())
    y = [results[k] for k in x]

    plt.figure()
    plt.plot(x, y, marker='o')
    plt.xlabel("Focus")
    plt.ylabel("Sharpness")
    plt.title("Focus Curve")
    plt.grid()
    plt.show()

# =============================
# 7. 显示图像
# =============================
def show_frame(frame, title=""):
    img = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# =============================
# 8. 三图对比（论文用）
# =============================
def show_comparison(best_frame, mid_frame, worst_frame):
    def norm(f):
        return cv2.normalize(f, None, 0, 255, cv2.NORM_MINMAX)

    plt.figure(figsize=(10,4))

    plt.subplot(1,3,1)
    plt.title("Worst Focus")
    plt.imshow(norm(worst_frame), cmap='gray')

    plt.subplot(1,3,2)
    plt.title("Medium")
    plt.imshow(norm(mid_frame), cmap='gray')

    plt.subplot(1,3,3)
    plt.title("Best Focus")
    plt.imshow(norm(best_frame), cmap='gray')

    plt.show()

# =============================
# 9. 主程序
# =============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["capture", "analyze", "analyze_csv"], default="capture")
    parser.add_argument("--folder", default="./event_data/")
    parser.add_argument("--duration-ms", type=int, default=CAPTURE_DURATION_MS)
    parser.add_argument("--frame-interval-ms", type=int, default=FRAME_INTERVAL_MS)
    parser.add_argument("--diopter-update-ms", type=int, default=DIOPTER_UPDATE_MS)
    parser.add_argument("--start-diopter", type=float, default=START_DIOPTER)
    parser.add_argument("--end-diopter", type=float, default=END_DIOPTER)
    parser.add_argument("--csv-dir", default=CSV_DIR)
    parser.add_argument("--csv-name", default=CSV_NAME)
    parser.add_argument("--csv-path", default="")
    parser.add_argument("--picture-dir", default=PICTURE_DIR)
    args = parser.parse_args()

    if args.mode == "capture":
        collect_focus_data(
            duration_ms=args.duration_ms,
            frame_interval_ms=args.frame_interval_ms,
            diopter_update_ms=args.diopter_update_ms,
            start_diopter=args.start_diopter,
            end_diopter=args.end_diopter,
            csv_dir=args.csv_dir,
            csv_name=args.csv_name,
            picture_dir=args.picture_dir,
        )
    elif args.mode == "analyze_csv":
        csv_path = args.csv_path if args.csv_path else os.path.join(args.csv_dir, args.csv_name)
        analyze_csv_focus(
            csv_path,
            duration_ms=args.duration_ms,
            frame_interval_ms=args.frame_interval_ms,
            start_diopter=args.start_diopter,
            end_diopter=args.end_diopter,
            picture_dir=args.picture_dir,
        )
    else:
        best_focus, worst_focus, results, frames, focus_times, focus_states = find_best_focus(args.folder)

        plot_focus_curve(results)

        show_frame(frames[best_focus], "Best Focus")
        show_frame(frames[worst_focus], "Worst Focus")

        sorted_focus = sorted(results.keys())
        mid_focus = sorted_focus[len(sorted_focus) // 2]

        show_comparison(
            frames[best_focus],
            frames[mid_focus],
            frames[worst_focus],
        )

        print("\n===== ELP 判定 =====")
        for focus_value in sorted(focus_times.keys()):
            t_mark = focus_times[focus_value]
            state = focus_states[focus_value]
            if t_mark is None:
                print(f"Focus {focus_value:.3f}: {state}, 无符号突变")
            else:
                print(f"Focus {focus_value:.3f}: {state}, 正好对焦时刻 t = {t_mark}")