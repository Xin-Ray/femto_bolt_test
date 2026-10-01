"""Headless camera check: device info, then the frame rates the camera delivers
for color alone, depth alone, and color + depth together (as ./run_depth.sh uses them).

    python scripts/check_camera.py [--seconds 5] [--width 0 --height 0]
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pyorbbecsdk import Config, Context, OBFrameAggregateOutputMode, OBSensorType, Pipeline

from charuco_demo import pick_color_profile


def measure(device, args, color, depth):
    """Frames per second received for each enabled stream (no decoding)."""
    pipeline = Pipeline(device)
    config = Config()
    if color:
        profile = pick_color_profile(pipeline, args.width, args.height, 30)
        config.enable_stream(profile)
        size = f"{profile.get_width()}x{profile.get_height()} {profile.get_format()}"
    if depth:
        config.enable_stream(pipeline.get_stream_profile_list(
            OBSensorType.DEPTH_SENSOR).get_default_video_stream_profile())
    config.set_frame_aggregate_output_mode(OBFrameAggregateOutputMode.DISABLE)

    pipeline.start(config)
    n_color = n_depth = 0
    t0 = time.time()
    try:
        while time.time() - t0 < args.seconds:
            frames = pipeline.wait_for_frames(1000)
            if frames is None:
                continue
            n_color += frames.get_color_frame() is not None
            n_depth += frames.get_depth_frame() is not None
    finally:
        pipeline.stop()
    parts = []
    if color:
        parts.append(f"color {n_color / args.seconds:5.1f} fps ({size})")
    if depth:
        parts.append(f"depth {n_depth / args.seconds:5.1f} fps")
    return "   ".join(parts)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seconds", type=float, default=5)
    p.add_argument("--width", type=int, default=0, help="color width (default 0 = largest available)")
    p.add_argument("--height", type=int, default=0)
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
                               ("color+depth", True, True)):
        print(f"{name:12s} {measure(device, args, color, depth)}")


if __name__ == "__main__":
    main()
