"""Headless camera check: device info, then frame rates for color, depth and aligned color+depth.

    ./run_demo.sh is the live viewer; this needs no display:
    python scripts/check_camera.py [--seconds 5]
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pyorbbecsdk import (AlignFilter, Config, Context, OBFrameAggregateOutputMode,
                         OBSensorType, OBStreamType, Pipeline)

from charuco_demo import color_to_bgr, pick_color_profile


def measure(device, seconds, color, depth):
    pipeline = Pipeline(device)
    config = Config()
    if color:
        config.enable_stream(pick_color_profile(pipeline, 1280, 720, 30))
    if depth:
        config.enable_stream(pipeline.get_stream_profile_list(
            OBSensorType.DEPTH_SENSOR).get_default_video_stream_profile())
    align = None
    if color and depth:
        config.set_frame_aggregate_output_mode(OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE)
        align = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)

    pipeline.start(config)
    got = 0
    t0 = time.time()
    try:
        while time.time() - t0 < seconds:
            frames = pipeline.wait_for_frames(1000)
            if frames is None:
                continue
            if align is not None:
                frames = align.process(frames)
                if not frames:
                    continue
                frames = frames.as_frame_set()
            if color and (frames.get_color_frame() is None
                          or color_to_bgr(frames.get_color_frame()) is None):
                continue
            if depth and frames.get_depth_frame() is None:
                continue
            got += 1
    finally:
        pipeline.stop()
    return got / seconds


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seconds", type=float, default=5)
    args = p.parse_args()

    ctx = Context()
    devices = ctx.query_devices()
    if devices.get_count() == 0:
        sys.exit("No Orbbec device found (lsusb -d 2bc5: should list it).")
    device = devices.get_device_by_index(0)
    info = device.get_device_info()
    print(f"device: {info.get_name()}  serial: {info.get_serial_number()}  "
          f"fw: {info.get_firmware_version()}  usb: {info.get_connection_type()}")

    for name, color, depth in (("color", True, False), ("depth", False, True),
                               ("color+depth aligned", True, True)):
        fps = measure(device, args.seconds, color, depth)
        print(f"{name:20s} {fps:5.1f} fps")


if __name__ == "__main__":
    main()
