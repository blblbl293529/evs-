# -----------------------------------------------------------------------------
# Copyright (c) 2025, Lucid Vision Labs, Inc.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
# BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Warning:
#
# EVS examples support only on windows at the moment
# -----------------------------------------------------------------------------

import time
import locale
import os
import ctypes
import csv
import threading
import numpy as np
import cv2

from arena_api.system import system
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

from lens import Lens

'''
Acquisition: EVS
    This example demonstrates how to acquire images using the EVS
    (Electronic Viewfinder System) stream protocol, which is designed
    to provide efficient and high-quality image transfer from the camera
    to the host system.
'''
TAB1 = "  "
TAB2 = "    "
TAB3 = "         "
'''
=-=-=-=-=-=-=-=-=-
=-=- EXAMPLE =-=-
=-=-=-=-=-=-=-=-=-
'''

def create_devices_with_tries():
        '''
        This function waits for the user to connect a device before raising
                an exception
        '''

        tries = 0
        tries_max = 6
        sleep_time_secs = 10
        while tries < tries_max:  # Wait for device for 60 seconds
                devices = system.create_device()
                if not devices:
                        print(
                                f'{TAB1}Try {tries+1} of {tries_max}: waiting for {sleep_time_secs} '
                                f'secs for a device to be connected!')
                        for sec_count in range(sleep_time_secs):
                                time.sleep(1)
                                print(f'{TAB1}{sec_count + 1 } seconds passed ',
                                        '.' * sec_count, end='\r')
                        tries += 1
                else:
                        print(f'{TAB1}Created {len(devices)} device(s)')
                        return devices
        else:
                raise Exception(f'{TAB1}No device found! Please connect a device and run '
                                                f'the example again.')


def get_evs_event_rate(rate):
    if rate < 1000:
        return f"{rate:.0f} ev/s"
    elif rate < 1000 * 1000:
        return f"{(rate / 1000):.1f} Kev/s"
    elif rate < 1000 * 1000 * 1000:
        return f"{(rate / (1000 * 1000)):.1f} Mev/s"
    else:
        return f"{(rate / (1000 * 1000 * 1000)):.1f} Gev/s"


def get_evs_gvsp_frame_rate(rate):
    return f"{rate:.0f} Bid/s"


def get_evs_link_throughput(rate):
    if rate < 1000:
        return f"{rate:.0f} Bps"
    elif rate < 1000 * 1000:
        return f"{(rate / 1000):.1f} KBps"
    elif rate < 1000 * 1000 * 1000:
        return f"{(rate / (1000 * 1000)):.1f} MBps"
    else:
        return f"{(rate / (1000 * 1000 * 1000)):.1f} GBps"


'''
demonstrates evs acquisition
    (1) sets acquisition mode
    (2) sets buffer handling mode
    (3) sets Event Format to EVS and camera event rate to 10 Mev/s
    (4) sets EVS output format to CDFrame
    (5) sets EVS accumulation time to auto
    (6) starts the stream
    (7) gets a number of images
    (8) prints information from images
    (9) requeues buffers
    (10) stops the stream
'''
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


def configure_and_get_image_buffers(device, lens=None):

        nodemap = device.nodemap
        tl_stream_nodemap = device.tl_stream_nodemap
        nodes = device.nodemap.get_node(['Width', 'Height'])

        width = nodes['Width'].value
        height= nodes['Height'].value
        save_dir = r"C:\Users\13119\OneDrive\Desktop\evs\Images"
        cdframe_dir = r"C:\Users\13119\OneDrive\Desktop\evs\data\cdframe"
        csv_dir = r"C:\Users\13119\OneDrive\Desktop\evs\data\csv"
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(cdframe_dir, exist_ok=True)
        os.makedirs(csv_dir, exist_ok=True)

        print(f'{TAB1}Image (w, h) = ({width} , {height} )')

        # Set features before streaming.-------------------------------------------
        print(f'{TAB1}Set acquisition mode to \'Continuous\'')

        initial_acquisition_mode = nodemap.get_node("AcquisitionMode").value

        nodemap.get_node("AcquisitionMode").value = "Continuous"

        print(f'{TAB1}Set buffer handling mode to \'NewestOnly\'')

        tl_stream_nodemap["StreamBufferHandlingMode"].value = "NewestOnly"

        '''
        The EventFormat node determines whether the camera can use the EVS datastream engine.
                When set to EVS, Arena switches to the EVS engine. If EVS is not supported,
                the acquisition mode is restored to its original setting, and the process is exited.
        '''

        try:
                event_format_initial = nodemap.get_node('EventFormat').value

                print(f'{TAB1}Set Event Format to EVT3.0')
                nodemap["EventFormat"].value = "EVT3_0"

                # set camera event rate to 10 Mev/s
                print(f'{TAB1}Set Camera Event Rate to 10 Mev/s')
                erc_enable_initial = nodemap.get_node('ErcEnable').value
                nodemap["ErcEnable"].value = True

                camera_event_rate_initial = nodemap.get_node('ErcRateLimit').value
                nodemap["ErcRateLimit"].value = 10.0

                # set evs output format to XYTPFrame (事件)，由主机生成 CDFrame
                print(f'{TAB1}Set EVS output format to XYTPFrame')
                tl_stream_nodemap["StreamEvsOutputFormat"].value = "XYTPFrame"

                number_of_buffers = 50
                device.start_stream(number_of_buffers)
                print(f'{TAB1}Stream started with {number_of_buffers} buffers')

                total_ms = 1000
                frame_interval_ms = 30
                next_frame_ms = frame_interval_ms
                frame_count = 0
                all_events = []

                accumulated_on = np.zeros((height, width), dtype=np.float32)
                accumulated_off = np.zeros((height, width), dtype=np.float32)
                total_on = np.zeros((height, width), dtype=np.float32)
                total_off = np.zeros((height, width), dtype=np.float32)

                stop_focusing = False

                def focus_thread():
                        if lens is None or not lens.is_connected():
                                return
                        try:
                                if not lens.mode == 5:
                                        lens.to_focal_power_mode()

                                phase = 0
                                last_diopter = None
                                while not stop_focusing:
                                        elapsed_ms = (time.time() - start_time) * 1000
                                        if elapsed_ms < 500:
                                                phase = 1
                                                target_diopter = -3
                                        elif elapsed_ms < 1000:
                                                if phase == 1:
                                                        print("[变焦线程] ========== 屈光度突变: -3 → 0 ==========")
                                                        phase = 2
                                                target_diopter = 0
                                        else:
                                                break

                                        if last_diopter != target_diopter:
                                                lens.set_diopter(target_diopter)
                                                print(f"[变焦线程] {elapsed_ms:.1f}ms - 设置屈光度: {target_diopter}")
                                                last_diopter = target_diopter

                                        time.sleep(0.01)
                                print("[变焦线程] 采集完成")
                        except Exception as e:
                                print(f"[变焦线程] 错误: {e}")

                if lens is not None and lens.is_connected():
                        try:
                                if not lens.mode == 5:
                                        lens.to_focal_power_mode()
                                lens.set_diopter(-3)
                                print("[变焦线程] 初始屈光度: -3")
                        except Exception as e:
                                print(f"[变焦线程] 初始化屈光度失败: {e}")

                start_time = time.time()

                focus_th = threading.Thread(target=focus_thread, daemon=True)
                focus_th.start()

                def save_frame(elapsed_ms):
                        nonlocal frame_count, accumulated_on, accumulated_off
                        diff = accumulated_on - accumulated_off
                        img_u8 = to_cdframe_uint8(diff)
                        save_path = os.path.join(cdframe_dir, f"cdframe_{frame_count:04d}.png")
                        cv2.imwrite(save_path, img_u8)
                        print(f"{TAB3}Saved: {save_path} ({elapsed_ms:.1f}ms)")
                        frame_count += 1
                        accumulated_on = np.zeros((height, width), dtype=np.float32)
                        accumulated_off = np.zeros((height, width), dtype=np.float32)

                try:
                        while (time.time() - start_time) * 1000 < total_ms:
                                buffer = device.get_buffer(timeout=5)

                                if buffer.is_incomplete:
                                        device.requeue_buffer(buffer)
                                        continue

                                events = parse_xytp_buffer(buffer, width, height)
                                if events:
                                        all_events.extend(events)
                                        for t, x, y, p in events:
                                                if p == 1:
                                                        accumulated_on[y, x] += 1.0
                                                        total_on[y, x] += 1.0
                                                else:
                                                        accumulated_off[y, x] += 1.0
                                                        total_off[y, x] += 1.0

                                elapsed_ms = (time.time() - start_time) * 1000
                                while elapsed_ms >= next_frame_ms and next_frame_ms < total_ms:
                                        save_frame(elapsed_ms)
                                        next_frame_ms += frame_interval_ms

                                device.requeue_buffer(buffer)
                finally:
                        stop_focusing = True
                        focus_th.join(timeout=1)

                device.stop_stream()
                print(f'{TAB1}Stream stopped')

                # 保存全时段累计事件彩色图（ON=红，OFF=蓝，背景=白）
                on_scale = float(np.percentile(total_on, 99)) if (total_on.size > 0) else 1.0
                off_scale = float(np.percentile(total_off, 99)) if (total_off.size > 0) else 1.0
                if on_scale <= 0:
                        on_scale = 1.0
                if off_scale <= 0:
                        off_scale = 1.0
                on_u8 = np.clip((total_on / on_scale) * 255.0, 0, 255).astype(np.uint8)
                off_u8 = np.clip((total_off / off_scale) * 255.0, 0, 255).astype(np.uint8)
                # 白底上混合纯红/纯蓝，避免黄/青
                total_rgb = np.full((height, width, 3), 255, dtype=np.float32)
                on_alpha = (on_u8.astype(np.float32) / 255.0)[..., None]
                off_alpha = (off_u8.astype(np.float32) / 255.0)[..., None]

                red = np.array([0, 0, 255], dtype=np.float32)
                blue = np.array([255, 0, 0], dtype=np.float32)

                total_rgb = total_rgb * (1.0 - on_alpha) + red * on_alpha
                total_rgb = total_rgb * (1.0 - off_alpha) + blue * off_alpha
                total_rgb = np.clip(total_rgb, 0, 255).astype(np.uint8)
                total_path = os.path.join(cdframe_dir, "cdframe_total.png")
                cv2.imwrite(total_path, total_rgb)
                print(f"{TAB2}累计事件图: {total_path}")

                if all_events:
                        all_events = sorted(all_events, key=lambda e: e[0])
                        csv_path = os.path.join(csv_dir, "events_txyp.csv")
                        with open(csv_path, 'w', newline='') as f:
                                writer = csv.writer(f)
                                writer.writerow(['t', 'x', 'y', 'p'])
                                writer.writerows(all_events)
                        print(f"{TAB2}CSV 保存: {csv_path}")
                        print(f"{TAB2}总事件数: {len(all_events):,}")
                else:
                        print(f"{TAB2}无事件数据")

                # Restore initial values
                nodemap.get_node("ErcEnable").value = erc_enable_initial
                nodemap.get_node("ErcRateLimit").value = camera_event_rate_initial
                nodemap.get_node("EventFormat").value = event_format_initial

        except:
                print(f'{TAB1}Connected camera does not support any EventFormats\n')

        # Restore initial values
        nodemap.get_node("AcquisitionMode").value = initial_acquisition_mode


def example_entry_point():

        # Get connected devices ---------------------------------------------------

        lens = Lens('COM3', debug=False)
        if lens.is_connected() and lens.camera_device is not None:
                device = lens.camera_device
        else:
                devices = create_devices_with_tries()
                device = system.select_device(devices)

        # Configure device and grab images ----------------------------------------

        configure_and_get_image_buffers(device, lens)

        # Clean up ----------------------------------------------------------------

        if lens is not None:
                lens.close_connection()
        else:
                system.destroy_device()
        print(f'{TAB1}Destroyed all created devices')


if __name__ == '__main__':
        print('\nExample started\n')
        example_entry_point()
        print('\nExample finished successfully')