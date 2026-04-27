import pandas as pd
import numpy as np
import tonic
import tonic.transforms as transforms
import matplotlib.pyplot as plt

def load_events_from_csv(file_path):
    """
    读取 CSV 并转换为 Tonic 兼容的 numpy 结构化数组
    """
    # 1. 读取 CSV
    # 如果你的 CSV 没有表头，请添加 header=None, names=['x', 'y', 't', 'p']
    df = pd.read_csv(file_path)
    
    # 确保列名正确，防止 CSV 里是 timestamp 或者 polarity
    # 如果名字不一样，请手动重命名
    # df = df.rename(columns={'timestamp': 't', 'polarity': 'p'})

    # 兼容不同列顺序（本项目默认 CSV 为 t,x,y,p）
    if set(['t', 'x', 'y', 'p']).issubset(df.columns):
        df = df[['x', 'y', 't', 'p']]

    # 2. 检查极性 (Polarity)
    # Tonic 要求 p 必须是 0 或 1。如果你的数据是 -1/1，需要转换
    if df['p'].min() < 0:
        print("检测到极性包含负数，正在转换为 0/1...")
        # 假设 -1 代表 OFF(0)，1 代表 ON(1)
        df['p'] = (df['p'] + 1) // 2 
    
    # 3. 检查时间单位
    # Tonic 通常处理微秒(us)。如果你的 t 是秒(如 0.001)，请乘以 1e6 并转为整数
    if df['t'].dtype.kind in ('f', 'i', 'u'):
        t_max = df['t'].max()
        # 简单判定：若时间跨度明显小于 1e6，按秒处理
        if t_max < 1e6:
            df['t'] = (df['t'] * 1e6).astype(np.int64)
        else:
            df['t'] = df['t'].astype(np.int64)

    # 4. 构建 Tonic 需要的结构化数组
    # dtype 必须严格匹配：x, y, t, p
    events = np.zeros(len(df), dtype=[('x', '<i4'), ('y', '<i4'), ('t', '<i8'), ('p', '<i4')])
    events['x'] = df['x'].values
    events['y'] = df['y'].values
    events['t'] = df['t'].values
    events['p'] = df['p'].values
    
    return events

# --- 主程序 ---

# 1. 加载数据
csv_file = r"C:\Users\13119\OneDrive\Desktop\evs\data\focus\csv\focus_csv.csv" # 替换为你的 CSV 路径
events = load_events_from_csv(csv_file)
W, H = 1280, 720

# 2. 定义转换器
# time_window: 多少微秒的数据合成一张图
transform = transforms.ToFrame(
    sensor_size=(W, H, 2), 
    time_window=30000  # 30ms
)

# 3. 转换
frames = transform(events)
print(f"生成的张量形状: {frames.shape}") # (帧数, 2, H, W)

# 4. 可视化与保存
if len(frames) > 0:
    import os
    save_dir = r"C:\Users\13119\OneDrive\Desktop\evs\Images"
    cdframe_dir = os.path.join(save_dir, "cdframe")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(cdframe_dir, exist_ok=True)

    # 合并所有帧为一张总事件帧
    merged_frame = np.sum(frames, axis=0)
    merged_img = merged_frame[1] - merged_frame[0]
    merged_img = np.clip(merged_img, -1, 1)
    merged_img_uint8 = ((merged_img + 1) * 127.5).astype(np.uint8)
    merged_save_path = os.path.join(save_dir, "events.png")
    plt.imsave(merged_save_path, merged_img_uint8, cmap='seismic', vmin=0, vmax=255)
    print(f"已保存合并事件帧: {merged_save_path}")

    def to_cdframe_uint8(diff, scale):
        """
        将 ON-OFF 差分映射到 8-bit 灰度：
        0.0 对应 128，正为亮，负为暗（更接近 Arena CDFrame）
        """
        if scale <= 0:
            scale = 1.0
        img = 128.0 + (diff / scale) * 127.0
        img = np.clip(img, 0, 255)
        return img.astype(np.uint8)

    # 统一缩放：用全体帧的 |diff| 的 99 分位作为幅度
    all_diff = np.abs(frames[:, 1, :, :] - frames[:, 0, :, :]).astype(np.float32)
    scale = float(np.percentile(all_diff, 99)) if all_diff.size > 0 else 1.0

    print(f"保存所有帧到 {save_dir} ...")
    for frame_idx, frame in enumerate(frames):
        img_display = frame[1] - frame[0]
        img_uint8 = to_cdframe_uint8(img_display, scale)
        save_path = os.path.join(save_dir, f"frame_{frame_idx:04d}.png")
        plt.imsave(save_path, img_uint8, cmap='seismic', vmin=0, vmax=255)

        # CDFrame 风格：灰度表示事件累积强度
        cdframe = frame[1] - frame[0]
        cdframe_u8 = to_cdframe_uint8(cdframe, scale)
        cdframe_path = os.path.join(cdframe_dir, f"cdframe_{frame_idx:04d}.png")
        plt.imsave(cdframe_path, cdframe_u8, cmap='gray', vmin=0, vmax=255)
    print(f"共保存 {len(frames)} 张图片到 {save_dir}")
    print(f"CDFrame 已保存到 {cdframe_dir}")
    # 可选：显示最后一帧做检查（默认关闭，避免阻塞）
    PREVIEW = False
    if PREVIEW:
        plt.imshow(img_uint8, cmap='seismic', vmin=0, vmax=255)
        plt.title('Last Frame Preview')
        plt.show()
else:
    print("没有生成帧，请检查 time_window 是否设置过小，或者 CSV 数据的时间跨度不够。")