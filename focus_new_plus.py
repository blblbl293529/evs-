"""
focus_new_plus.py  ELP 符号翻转自动对焦系统（自参考版）
==========================================================

流程
----
1. 在线采集
   - ETL 从 START_DIOPTER 单向扫描至 END_DIOPTER（CAPTURE_DURATION_MS ms）
   - 事件相机持续记录事件流 e_i=(x_i,y_i,t_i,p_i)
   - 同步记录屈光度日志 d(t)
   - 保存 events.csv + diopter_log.csv

2. 离线分析
   a. 参考帧  将所有采集到的事件累积为无符号计数图，
              归一化后得到灰度参考图 I(x,y)，
              计算 L(x,y) = ∇²( GaussianBlur(I) )
              （该参考图不作为最终成像结果，只提供场景边缘/纹理结构）

   b. ELP 序列  以 Δt=FRAME_WINDOW_MS（10 ms）切割时间窗口：
              E⁺(x,y,t) = 窗口内正事件（亮度增加）计数图
              E⁻(x,y,t) = 窗口内负事件（亮度降低）计数图
              E(x,y,t)  = E⁺ - E⁻  （有符号事件帧）
              ELP(t)    = -∑_{(x,y)} L(x,y) · E(x,y,t)

   c. 符号翻转检测  寻找 ELP(t_i)·ELP(t_{i+1}) < 0，并满足：
              - 两侧窗口事件数 ≥ MIN_EVENTS_FRAME
              - 两侧 |ELP| ≥ SIGN_AMP_RATIO × max|ELP|
              - 斜率 |ΔELP/Δt| ≥ SIGN_SLOPE_MIN
              零交叉线性插值得到焦点时刻 t*

   d. 输出
              focus_time    = t*
              focus_diopter = d*(t*)  （屈光度日志线性插值）
              ELP 曲线图（标注符号翻转位置）
              elp_series.csv
"""

from __future__ import annotations

import csv
import ctypes
import json
import os
import struct
import threading
import time

import cv2
import matplotlib
try:
    matplotlib.use("TkAgg")
except Exception:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from arena_api.system import system
from lens import Lens

# ======================================================================
# 一、全局参数
# ======================================================================

# 相机
IMG_H = 720
IMG_W = 1280

# 采集 — 单向扫描
CAPTURE_DURATION_MS = 2000
DIOPTER_UPDATE_MS   = 6
START_DIOPTER       = -4.0
END_DIOPTER         =  4.0

# 参考帧 Laplacian（用扫焦累积帧离线计算）
LAP_MEDIAN_KSIZE = 0    # MedianBlur 核大小（0=跳过，避免把稀疏事件参考图抹平）
LAP_BLUR_KSIZE   = 5    # GaussianBlur 核大小（奇数）
LAP_BLUR_SIGMA   = 1.0  # GaussianBlur σ
LAP_KSIZE        = 3    # Laplacian 核大小

# ELP 时间窗口
FRAME_WINDOW_MS  = 6 # Δt (ms)，6ms 聚合保证稀疏场景下每窗口有足够事件
MIN_EVENTS_FRAME = 20    # 窗口内最少事件数（否则 ELP 视为无效）

# 符号翻转检测
SIGN_AMP_RATIO         = 0.02  # 两侧 |ELP| 均须 ≥ max|ELP| × ratio
SIGN_SLOPE_RATIO_PER_MS = 0.0  # 归一化斜率门限：|ΔELP|/(max_amp·Δt_ms) ≥ 此值（0=不约束）
SIGN_MIN_GAP_MS        = 12.0  # 相邻翻转最小间隔 (ms)（去重）

# 图像渲染
EVENT_RENDER_PERCENTILE = 99
EVENT_RENDER_GAMMA      = 0.6

# 输出目录
OUT_DIR = r"C:\Users\13119\OneDrive\Desktop\evs\data\focus\focus_new_plus"

# 相机 bias（负值 = 降低触发阈值 = 提高灵敏度）
ENABLE_BIAS_PRESET = True
BIAS_DELTAS = {
    "BiasDiffOn":        -5,
    "BiasDiffOff":       -5,
    "BiasRefractory":     0,
    "BiasHpf":            0,
    "BiasFo":             0,
    "EventThreshold":    -3,
    "EventThresholdOn":  -3,
    "EventThresholdOff": -3,
}

# ======================================================================
# 二、硬件工具函数
# ======================================================================

def _diopter_at_time_ms(t_ms: float, start_d: float, end_d: float,
                         dur_ms: float) -> float:
    if dur_ms <= 0:
        return start_d
    return start_d + (end_d - start_d) * max(0.0, min(t_ms / dur_ms, 1.0))


def _render_event_rgb(on: np.ndarray, off: np.ndarray,
                      percentile: float = EVENT_RENDER_PERCENTILE,
                      gamma: float | None = EVENT_RENDER_GAMMA) -> np.ndarray:
    """将 on/off 事件平面渲染为 BGR 图像（红=ON，蓝=OFF，白=背景）"""
    def _scale(plane):
        s = float(np.percentile(plane, percentile)) if plane.max() > 0 else 1.0
        return np.clip(plane / max(s, 1e-9) * 255.0, 0, 255).astype(np.uint8)

    on_u8  = _scale(on)
    off_u8 = _scale(off)
    if gamma is not None:
        def _g(x): return (np.clip(x.astype(np.float32) / 255.0, 0, 1) ** gamma * 255).astype(np.uint8)
        on_u8 = _g(on_u8); off_u8 = _g(off_u8)

    rgb = np.full((*on.shape, 3), 255, dtype=np.float32)
    oa  = (on_u8.astype(np.float32)  / 255.0)[..., None]
    fa  = (off_u8.astype(np.float32) / 255.0)[..., None]
    rgb = rgb * (1 - oa) + np.array([0, 0, 255], np.float32) * oa   # ON  → 红
    rgb = rgb * (1 - fa) + np.array([255, 0, 0], np.float32) * fa   # OFF → 蓝
    return np.clip(rgb, 0, 255).astype(np.uint8)


def _set_diopter(lens: Lens, d: float) -> None:
    if lens is None or not lens.is_connected():
        return
    if lens.mode != 5:
        lens.to_focal_power_mode()
    raw = int((d + 5) * 200) if lens.firmware_type == "A" else int(d * 200)
    lens.send_command(b"PwDA" + struct.pack(">h", raw) + b"\x00\x00")


def _parse_buffer(buf, width: int, height: int) -> list:
    ch  = 4
    bpp = buf.bits_per_pixel // 8
    n   = buf.size_filled // bpp
    if n == 0:
        return []
    src = ctypes.cast(buf.pdata, ctypes.POINTER(ctypes.c_float))
    events = []
    for i in range(n):
        o = i * ch
        x = int(round(src[o]))
        y = int(round(src[o + 1]))
        t = src[o + 2]
        p = 1 if src[o + 3] > 0.5 else 0
        if 0 <= x < width and 0 <= y < height:
            events.append([t, x, y, p])
    return events


def _apply_bias(nodemap) -> None:
    for name, delta in BIAS_DELTAS.items():
        try:
            node = nodemap.get_node(name)
            cur  = node.value
            if isinstance(cur, bool):
                continue
            lo = getattr(node, "min", cur - 999)
            hi = getattr(node, "max", cur + 999)
            node.value = max(lo, min(hi, cur + delta))
        except Exception:
            pass


def _configure_camera(device) -> None:
    nm = device.nodemap
    tl = device.tl_stream_nodemap
    nm["AcquisitionMode"].value          = "Continuous"
    tl["StreamBufferHandlingMode"].value = "OldestFirstOverwrite"
    nm["EventFormat"].value              = "EVT3_0"
    nm["ErcEnable"].value                = False
    nm["ErcRateLimit"].value             = 10.0
    tl["StreamEvsOutputFormat"].value    = "XYTPFrame"
    if ENABLE_BIAS_PRESET:
        _apply_bias(nm)


def _create_devices():
    for attempt in range(6):
        devices = system.create_device()
        if devices:
            return devices
        print(f"[camera] 第 {attempt+1}/6 次未找到设备，等待 10s …")
        time.sleep(10)
    raise RuntimeError("未找到相机设备")


# ======================================================================
# 三、ELP 时间序列
# ======================================================================

# ======================================================================
# 五、符号翻转检测
# ======================================================================

def build_reference(
    all_events: list,
    width: int,
    height: int,
    out_dir: str,
) -> np.ndarray:
    """
    Save three images and return the fixed Laplacian reference for ELP.
    """
    os.makedirs(out_dir, exist_ok=True)

    acc_on = np.zeros((height, width), dtype=np.float32)
    acc_off = np.zeros((height, width), dtype=np.float32)
    if all_events:
        ev = np.array(all_events, dtype=np.float64)
        xs = ev[:, 1].astype(np.int32)
        ys = ev[:, 2].astype(np.int32)
        ps = ev[:, 3].astype(np.int8)
        valid = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
        xs, ys, ps = xs[valid], ys[valid], ps[valid]
        np.add.at(acc_on, (ys[ps == 1], xs[ps == 1]), 1.0)
        np.add.at(acc_off, (ys[ps == 0], xs[ps == 0]), 1.0)

    color_img = _render_event_rgb(acc_on, acc_off)
    color_path = os.path.join(out_dir, "accumulated_event_color.png")
    cv2.imwrite(color_path, color_img)

    diff = acc_on - acc_off
    abs_diff = np.abs(diff)
    signed_scale = float(np.percentile(abs_diff[abs_diff > 0], 99)) if np.any(abs_diff > 0) else 1.0
    gray_img = np.clip(128.0 + diff / max(signed_scale, 1e-9) * 127.0, 0, 255).astype(np.uint8)
    gray_path = os.path.join(out_dir, "accumulated_event_gray.png")
    cv2.imwrite(gray_path, gray_img)

    total = acc_on + acc_off
    nonzero = total[total > 0]
    ref_scale = float(np.percentile(nonzero, 99)) if nonzero.size else 1.0
    reference = np.clip(total / max(ref_scale, 1e-9) * 255.0, 0, 255).astype(np.uint8)
    reference = cv2.dilate(reference, np.ones((3, 3), np.uint8), iterations=1)
    reference = cv2.GaussianBlur(reference, (LAP_BLUR_KSIZE, LAP_BLUR_KSIZE), LAP_BLUR_SIGMA)
    ref_path = os.path.join(out_dir, "Reference image.png")
    cv2.imwrite(ref_path, reference)

    L_ref = cv2.Laplacian(reference.astype(np.float32), cv2.CV_32F, ksize=LAP_KSIZE)
    np.save(os.path.join(out_dir, "ref_laplacian.npy"), L_ref)
    print("[ref] saved:")
    print(f"  color:     {color_path}")
    print(f"  gray:      {gray_path}")
    print(f"  reference: {ref_path}")
    print(f"[ref] Laplacian range=[{L_ref.min():.3f}, {L_ref.max():.3f}] "
          f"energy={float(np.sum(L_ref ** 2)):.3e}")
    return L_ref


def build_elp_series(
    all_events: list,
    L_ref: np.ndarray,
    t0_ms: float,
    duration_ms: float,
    window_ms: float = FRAME_WINDOW_MS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Paper-style ELP:
      ELP(t) = -sum(L_ref * E_frame(t)), where E_frame = E_on - E_off.
    """
    height, width = L_ref.shape
    n_frames = max(1, int(duration_ms / window_ms))

    ev_sorted = sorted(all_events, key=lambda e: e[0])
    ev_arr = np.array(ev_sorted, dtype=np.float64)
    ev_t = ev_arr[:, 0]
    ev_x = ev_arr[:, 1].astype(np.int32)
    ev_y = ev_arr[:, 2].astype(np.int32)
    ev_p = ev_arr[:, 3].astype(np.int8)

    t_centers_ms = np.empty(n_frames, dtype=np.float64)
    elp_arr = np.zeros(n_frames, dtype=np.float64)
    count_arr = np.zeros(n_frames, dtype=np.int64)

    for k in range(n_frames):
        t_lo = t0_ms + k * window_ms
        t_hi = t_lo + window_ms
        t_centers_ms[k] = t_lo + 0.5 * window_ms

        i_lo = int(np.searchsorted(ev_t, t_lo))
        i_hi = int(np.searchsorted(ev_t, t_hi))
        if i_hi <= i_lo:
            continue

        count_arr[k] = i_hi - i_lo
        xs = ev_x[i_lo:i_hi]
        ys = ev_y[i_lo:i_hi]
        ps = ev_p[i_lo:i_hi]

        E_frame = np.zeros((height, width), dtype=np.float32)
        mask_on = ps == 1
        mask_off = ps == 0
        if mask_on.any():
            np.add.at(E_frame, (ys[mask_on], xs[mask_on]), 1.0)
        if mask_off.any():
            np.add.at(E_frame, (ys[mask_off], xs[mask_off]), -1.0)

        elp_arr[k] = -float(np.sum(L_ref * E_frame))

    return t_centers_ms, elp_arr, count_arr


def detect_sign_reversals(
    t_arr: np.ndarray,
    elp_arr: np.ndarray,
    count_arr: np.ndarray,
) -> list[dict]:
    """
    在 ELP 曲线上检测可靠符号翻转（零交叉）。

    筛选条件
    --------
    a. 两侧窗口事件数均 ≥ MIN_EVENTS_FRAME
    b. 两侧 |ELP| ≥ SIGN_AMP_RATIO × max|ELP|（幅值门限）
    c. |斜率| ≥ SIGN_SLOPE_MIN（斜率门限）
    d. 相邻翻转间距 ≥ SIGN_MIN_GAP_MS（去重，保留 |slope| 最大者）

    零交叉插值
    ----------
    α  = -ELP[i] / (ELP[i+1] - ELP[i])
    t* = t[i] + α · Δt

    返回 list[dict]，每项含
    ----------------------
    t_star_ms   焦点时刻 (ms)
    elp_before  ELP[i]
    elp_after   ELP[i+1]
    slope       (ELP[i+1]-ELP[i]) / Δt
    direction   'pos_to_neg' 或 'neg_to_pos'
    """
    results: list[dict] = []
    if len(elp_arr) < 2:
        return results

    valid     = count_arr >= MIN_EVENTS_FRAME
    max_amp   = float(np.abs(elp_arr[valid]).max()) if valid.any() else 1.0
    amp_thr   = SIGN_AMP_RATIO * max_amp if max_amp > 0 else 0.0
    dt_ms     = float(t_arr[1] - t_arr[0]) if len(t_arr) > 1 else FRAME_WINDOW_MS

    for i in range(len(elp_arr) - 1):
        a, b = elp_arr[i], elp_arr[i + 1]
        if a * b >= 0:
            continue
        if count_arr[i] < MIN_EVENTS_FRAME or count_arr[i + 1] < MIN_EVENTS_FRAME:
            continue
        if abs(a) < amp_thr or abs(b) < amp_thr:
            continue
        # 归一化斜率：|ΔELP| / (max_amp · Δt_ms)，量纲为 1/ms
        slope_norm = abs(b - a) / max(max_amp, 1e-12) / dt_ms
        if slope_norm < SIGN_SLOPE_RATIO_PER_MS:
            continue
        slope = (b - a) / dt_ms   # 原始斜率（保留用于排序/记录）

        alpha  = -a / (b - a)
        t_star = float(t_arr[i]) + alpha * dt_ms
        results.append({
            "t_star_ms":  t_star,
            "elp_before": float(a),
            "elp_after":  float(b),
            "slope":      float(slope),
            "slope_norm": float(slope_norm),
            "direction":  "pos_to_neg" if a > 0 else "neg_to_pos",
        })

    # 去重：间距 < SIGN_MIN_GAP_MS 保留 |slope| 最大者
    if len(results) <= 1:
        return results
    merged: list[dict] = [results[0]]
    for r in results[1:]:
        if r["t_star_ms"] - merged[-1]["t_star_ms"] < SIGN_MIN_GAP_MS:
            if abs(r["slope"]) > abs(merged[-1]["slope"]):
                merged[-1] = r
        else:
            merged.append(r)
    return merged


# ======================================================================
# 六、屈光度插值
# ======================================================================

def interp_diopter(t_star_ms: float, diopter_log: list) -> float:
    if len(diopter_log) < 2:
        return _diopter_at_time_ms(t_star_ms, START_DIOPTER, END_DIOPTER,
                                    CAPTURE_DURATION_MS)
    ts = np.array([x[0] for x in diopter_log], dtype=np.float64)
    ds = np.array([x[1] for x in diopter_log], dtype=np.float64)
    return float(np.interp(t_star_ms, ts, ds))


# ======================================================================
# 七、可视化
# ======================================================================

def visualize_elp(
    t_arr: np.ndarray,
    elp_arr: np.ndarray,
    count_arr: np.ndarray,
    diopter_log: list,
    crossings: list[dict],
    out_dir: str,
) -> None:
    """
    三联图（共享时间轴）：
    ① ELP(t) 曲线 + 符号翻转标注
    ② 屈光度 d(t) + 翻转时刻标注
    ③ 每窗口事件数
    """
    os.makedirs(out_dir, exist_ok=True)
    clr = {"pos_to_neg": "red", "neg_to_pos": "limegreen"}

    # 将每窗口时刻映射到屈光度（用于图③ x 轴）
    if diopter_log and len(diopter_log) >= 2:
        dl_t_np = np.array([x[0] for x in diopter_log], dtype=np.float64)
        dl_d_np = np.array([x[1] for x in diopter_log], dtype=np.float64)
        d_for_count = np.interp(t_arr, dl_t_np, dl_d_np)
    else:
        d_for_count = np.array(
            [_diopter_at_time_ms(float(t), START_DIOPTER, END_DIOPTER,
                                  CAPTURE_DURATION_MS) for t in t_arr],
            dtype=np.float64)

    fig, axes = plt.subplots(3, 1, figsize=(13, 10))

    # ── ① ELP 曲线（x 轴：时间）─────────────────────────────────
    ax = axes[0]
    ax.plot(t_arr, elp_arr, lw=0.9, color="royalblue", label="ELP(t)")
    ax.axhline(0, color="gray", lw=0.8, ls="--", alpha=0.7)
    for cr in crossings:
        c = clr.get(cr["direction"], "orange")
        ax.axvline(cr["t_star_ms"], color=c, lw=2.0,
                   label=f"t*={cr['t_star_ms']:.1f}ms ({cr['direction']})")
        ax.scatter([cr["t_star_ms"]], [0], color=c, s=70, zorder=5)
    ax.set_xlabel("Wall-clock elapsed (ms)")
    ax.set_ylabel("ELP(t)")
    ax.set_title("① ELP curve — sign reversal detection (red: pos→neg, green: neg→pos)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)

    # ── ② 屈光度随时间（x 轴：时间）────────────────────────────
    ax = axes[1]
    if diopter_log:
        dl_t = np.array([x[0] for x in diopter_log])
        dl_d = np.array([x[1] for x in diopter_log])
        ax.plot(dl_t, dl_d, lw=1.2, color="darkorange", label="ETL d(t) [feedback]")
    for cr in crossings:
        d_star = interp_diopter(cr["t_star_ms"], diopter_log)
        c = clr.get(cr["direction"], "orange")
        ax.axvline(cr["t_star_ms"], color=c, lw=1.8, ls="--", alpha=0.8)
        ax.scatter([cr["t_star_ms"]], [d_star], color=c, s=70, zorder=5,
                   label=f"d*={d_star:+.3f} D")
    ax.set_xlabel("Wall-clock elapsed (ms)")
    ax.set_ylabel("Diopter (D)")
    ax.set_title("② Diopter vs time (ETL hardware feedback)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # ── ③ 每窗口事件数 vs 屈光度（x 轴：屈光度）────────────────
    ax = axes[2]
    # 排序保证 x 单调（单向扫描下 d 单调递增）
    order   = np.argsort(d_for_count)
    d_plot  = d_for_count[order]
    cnt_plot = count_arr[order]
    bar_w = (d_plot[-1] - d_plot[0]) / len(d_plot) * 0.9 if len(d_plot) > 1 else 0.1
    ax.bar(d_plot, cnt_plot, width=bar_w, color="steelblue", alpha=0.7)
    ax.axhline(MIN_EVENTS_FRAME, color="red", lw=0.8, ls="--",
               label=f"MIN_EVENTS={MIN_EVENTS_FRAME}")
    for cr in crossings:
        d_star = interp_diopter(cr["t_star_ms"], diopter_log)
        c = clr.get(cr["direction"], "orange")
        ax.axvline(d_star, color=c, lw=1.5, ls="--", alpha=0.8)
    ax.set_xlabel("Diopter (D)")
    ax.set_ylabel("Events / window")
    ax.set_title("③ Events per window vs diopter")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    save_path = os.path.join(out_dir, "elp_analysis.png")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[vis] 保存: {save_path}")


# ======================================================================
# 八、在线采集（仅扫焦 + 存 CSV）
# ======================================================================

def collect_focus_data(out_dir: str = OUT_DIR) -> tuple[int, int]:
    """
    在线阶段：ETL 单向扫焦 + 事件采集，保存 events.csv + diopter_log.csv。
    返回 (width, height)。
    """
    lens   = Lens("COM3", debug=False)
    devs   = _create_devices()
    device = system.select_device(devs)
    width  = device.nodemap["Width"].value
    height = device.nodemap["Height"].value

    os.makedirs(out_dir, exist_ok=True)
    _configure_camera(device)

    all_events:  list = []

    try:
        print(f"[collect] 扫焦: {START_DIOPTER:+.1f} D → {END_DIOPTER:+.1f} D  "
              f"历时 {CAPTURE_DURATION_MS} ms")

        # ETL 移至起始位置，等待稳定后读取实际反馈值
        _set_diopter(lens, START_DIOPTER)
        time.sleep(0.3)
        try:
            d_init = lens.get_diopter()
        except Exception:
            d_init = START_DIOPTER
        diopter_log: list = [(0.0, d_init)]
        print(f"[collect] 初始屈光度（反馈）: {d_init:+.3f} D")

        # ETL 稳定后再启动采流，避免预扫阶段事件污染 buffer
        device.start_stream(100)
        t_start    = time.time()
        t0_wall_ms: float = -1.0   # 首批事件到达的 wall-clock 偏移(ms)
        stop_flag  = [False]

        def _lens_thread():
            while not stop_flag[0]:
                elap = (time.time() - t_start) * 1000.0
                if elap >= CAPTURE_DURATION_MS:
                    break
                d_cmd = _diopter_at_time_ms(elap, START_DIOPTER, END_DIOPTER,
                                             CAPTURE_DURATION_MS)
                try:
                    _set_diopter(lens, d_cmd)
                    # 读取硬件反馈值（比命令值更准确，含 ETL 滞后）
                    try:
                        d_fb = lens.get_diopter()
                    except Exception:
                        d_fb = d_cmd   # 读取失败时退回命令值
                    diopter_log.append((elap, d_fb))
                except Exception as e:
                    print(f"[lens] {e}")
                time.sleep(max(DIOPTER_UPDATE_MS, 1) / 1000.0)

        th = threading.Thread(target=_lens_thread, daemon=True)
        th.start()

        total_timeouts = 0
        total_bufs     = 0
        t_last_wall_ms = -1.0
        next_diag_ms   = 200.0  # 每 200 ms 打印一次进度，确认相机在持续工作

        while (time.time() - t_start) * 1000.0 < CAPTURE_DURATION_MS:
            elap_ms = (time.time() - t_start) * 1000.0
            if elap_ms >= next_diag_ms:
                print(f"[buf] {elap_ms:6.0f} ms | events={len(all_events):,}  "
                      f"bufs={total_bufs}  timeouts={total_timeouts}")
                next_diag_ms += 200.0
            try:
                buf = device.get_buffer(timeout=5)
            except Exception:
                total_timeouts += 1
                time.sleep(0.005)
                continue
            total_bufs += 1
            if buf.is_incomplete:
                device.requeue_buffer(buf)
                continue
            evts = _parse_buffer(buf, width, height)
            if evts:
                curr_wall = (time.time() - t_start) * 1000.0
                if t0_wall_ms < 0:
                    t0_wall_ms = curr_wall
                t_last_wall_ms = curr_wall
                all_events.extend([evt + [curr_wall] for evt in evts])
            device.requeue_buffer(buf)

        stop_flag[0] = True
        th.join(timeout=1.5)
        if t0_wall_ms    < 0: t0_wall_ms    = 0.0
        if t_last_wall_ms < 0: t_last_wall_ms = 0.0
        print(f"[collect] 完成: {len(all_events):,} events  "
              f"bufs={total_bufs}  timeouts={total_timeouts}")
        print(f"[collect] 事件 wall-clock 跨度: "
              f"{t0_wall_ms:.1f} ~ {t_last_wall_ms:.1f} ms  "
              f"(有事件 {t_last_wall_ms - t0_wall_ms:.1f} ms)")

    finally:
        try:
            device.stop_stream()
        except Exception:
            pass
        try:
            system.destroy_device()
        except Exception:
            pass

    csv_events = os.path.join(out_dir, "events.csv")
    with open(csv_events, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_us", "x", "y", "p", "host_t_ms"])
        w.writerows(all_events)
    print(f"[collect] 事件 CSV: {csv_events}")

    csv_dlog = os.path.join(out_dir, "diopter_log.csv")
    with open(csv_dlog, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["elapsed_ms", "diopter"])
        w.writerows(diopter_log)
    print(f"[collect] 屈光度日志: {csv_dlog}")

    meta_file = os.path.join(out_dir, "meta.json")
    with open(meta_file, "w") as f:
        json.dump({
            "t0_wall_ms":     t0_wall_ms,
            "t_last_wall_ms": t_last_wall_ms,
            "total_timeouts": total_timeouts,
            "total_bufs":     total_bufs,
        }, f, indent=2)
    print(f"[collect] 时间对齐元数据: {meta_file}")

    return width, height


# ======================================================================
# 九、离线分析
# ======================================================================

def analyze_from_csv(
    out_dir: str = OUT_DIR,
    width:   int = IMG_W,
    height:  int = IMG_H,
) -> list[dict]:
    """
    离线阶段：
    events.csv + diopter_log.csv
    → 累积参考帧 + Laplacian
    → ELP 序列
    → 符号翻转检测
    → focus_time / focus_diopter

    返回所有检测到的符号翻转列表（含 focus_diopter 字段）。
    """
    import pandas as pd

    csv_events = os.path.join(out_dir, "events.csv")
    csv_dlog   = os.path.join(out_dir, "diopter_log.csv")
    for p in [csv_events, csv_dlog]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"找不到: {p}")

    # ── 加载数据 ──────────────────────────────────────────────────
    df_ev = pd.read_csv(csv_events).rename(
        columns={"t_us": "t", "timestamp": "t"})
    sensor_t = df_ev["t"].to_numpy(dtype=np.float64)
    if "host_t_ms" in df_ev.columns:
        # 新数据：使用电脑接收到事件时的 elapsed ms，与 diopter_log 同一时间轴。
        df_ev["elp_t_ms"] = df_ev["host_t_ms"].astype(np.float64)
        print("[align] ELP 分析时间轴: host_t_ms（推荐）")
    else:
        # 旧数据兼容：没有 host_t_ms 时只能退回 sensor timestamp。
        df_ev["elp_t_ms"] = (sensor_t - sensor_t.min()) / 1000.0
        print("[align] ELP 分析时间轴: sensor timestamp fallback（旧 CSV）")
    all_events = df_ev[["elp_t_ms", "x", "y", "p"]].values.tolist()
    all_events.sort(key=lambda e: e[0])
    print(f"[analyze] 事件: {len(all_events):,} 条")

    df_dl = pd.read_csv(csv_dlog)
    diopter_log = list(zip(df_dl["elapsed_ms"].tolist(),
                           df_dl["diopter"].tolist()))

    if not all_events:
        print("[analyze] 无事件，退出")
        return []

    # ── 加载时间对齐元数据 ────────────────────────────────────────
    meta_file      = os.path.join(out_dir, "meta.json")
    t0_wall_ms     = 0.0
    t_last_wall_ms = None   # type: float | None
    total_timeouts = None   # type: int | None
    total_bufs     = None   # type: int | None
    if os.path.exists(meta_file):
        with open(meta_file) as _f:
            _m = json.load(_f)
        t0_wall_ms     = float(_m.get("t0_wall_ms", 0.0))
        t_last_wall_ms = _m.get("t_last_wall_ms")
        total_timeouts = _m.get("total_timeouts")
        total_bufs     = _m.get("total_bufs")
    print(f"[align] 首批事件 wall-clock 偏移: {t0_wall_ms:.1f} ms")
    if t_last_wall_ms is not None:
        print(f"[align] 末批事件 wall-clock 时刻: {t_last_wall_ms:.1f} ms  "
              f"(wall-clock 事件窗: {t_last_wall_ms - t0_wall_ms:.1f} ms)")
    if total_timeouts is not None:
        print(f"[align] buffer总数={total_bufs}  超时次数={total_timeouts}  "
              f"估计无事件静默≈{total_timeouts * 5:.0f} ms")

    # ── 对齐检查：传感器时间跨度 vs 配置采集时长 ─────────────────
    t0_us = float(sensor_t.min())
    t_last_us = float(sensor_t.max())
    t_span_ms = (t_last_us - t0_us) / 1000.0
    print(f"[align] 事件传感器时间跨度: {t_span_ms:.1f} ms  "
          f"配置采集时长: {CAPTURE_DURATION_MS} ms  "
          f"差值: {abs(t_span_ms - CAPTURE_DURATION_MS):.1f} ms")
    if t_span_ms < CAPTURE_DURATION_MS * 0.5:
        wc_span = ((t_last_wall_ms - t0_wall_ms)
                   if t_last_wall_ms is not None else t_span_ms)
        d_s = _diopter_at_time_ms(
            t0_wall_ms, START_DIOPTER, END_DIOPTER, CAPTURE_DURATION_MS)
        d_e = _diopter_at_time_ms(
            t0_wall_ms + wc_span, START_DIOPTER, END_DIOPTER, CAPTURE_DURATION_MS)
        print(f"[align] event diopter range: {d_s:+.2f} D -> {d_e:+.2f} D  "
              f"(wall-clock span {wc_span:.1f} ms)")
        if wc_span < CAPTURE_DURATION_MS * 0.5:
            print("[align] WARNING: wall-clock event span also short "
                  "- possible buffer loss, check timeouts/bufs ratio")
        else:
            print("[align] sensor timestamp span is short "
                  "(normal for XYTPFrame batched timestamps; wall-clock OK)")

    # ── 步骤 a：ELP 时间序列（逐窗口 diff→blur→lap，host wall-clock ms）──
    L_ref = build_reference(all_events, width, height, out_dir)
    print("[analyze] ELP formula: -sum(L_ref * E_frame), fixed Reference image Laplacian")

    _t0_ms   = float(all_events[0][0])
    _tend_ms = float(all_events[-1][0])
    t_arr, elp_arr, count_arr = build_elp_series(
        all_events, L_ref, _t0_ms, _tend_ms - _t0_ms, FRAME_WINDOW_MS
    )
    n_valid = int((count_arr >= MIN_EVENTS_FRAME).sum())
    print(f"[analyze] 总窗口: {len(t_arr)}  有效窗口: {n_valid}")

    # ── 步骤 c：符号翻转检测 ─────────────────────────────────────
    crossings = detect_sign_reversals(t_arr, elp_arr, count_arr)
    print(f"[analyze] 检测到 {len(crossings)} 个有效符号翻转")

    # ── 步骤 d：屈光度插值 & 汇报 ────────────────────────────────
    results: list[dict] = []
    for cr in crossings:
        d_star = interp_diopter(cr["t_star_ms"], diopter_log)
        cr["focus_diopter"] = d_star
        results.append(cr)
        print(f"  focus_time = {cr['t_star_ms']:.2f} ms  "
              f"focus_diopter = {d_star:+.4f} D  "
              f"方向 = {cr['direction']}  |slope| = {abs(cr['slope']):.3f}")

    if not results:
        print("[analyze] 未检测到有效符号翻转，"
              "请检查：① 事件数量 ② BIAS 灵敏度 ③ SIGN_AMP_RATIO 阈值")
    else:
        best = max(results, key=lambda r: abs(r["slope"]))
        print(f"\n[analyze] 最显著翻转 →  "
              f"focus_time = {best['t_star_ms']:.2f} ms  "
              f"focus_diopter = {best['focus_diopter']:+.4f} D")

    # ── 保存 ELP 序列 CSV ────────────────────────────────────────
    elp_csv = os.path.join(out_dir, "elp_series.csv")
    with open(elp_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_center_ms", "elp", "event_count"])
        for i in range(len(t_arr)):
            w.writerow([round(float(t_arr[i]), 4),
                        round(float(elp_arr[i]), 8),
                        int(count_arr[i])])
    print(f"[analyze] ELP 序列 CSV: {elp_csv}")

    # ── 可视化 ────────────────────────────────────────────────────
    visualize_elp(t_arr, elp_arr, count_arr, diopter_log, crossings, out_dir)

    return results


if __name__ == "__main__":
    w, h = collect_focus_data(OUT_DIR)
    results = analyze_from_csv(OUT_DIR, width=w, height=h)
    if results:
        best = max(results, key=lambda r: abs(r["slope"]))
        print(f"\n最优焦平面：")
        print(f"  focus_time    = {best['t_star_ms']:.2f} ms")
        print(f"  focus_diopter = {best['focus_diopter']:+.4f} D")
