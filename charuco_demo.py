"""Live ChArUco detection + board pose on an Orbbec Femto Bolt.

Uses the camera's factory color intrinsics, so no calibration step is needed
to get a metric board pose. With --depth, the depth stream is aligned to color
and the measured depth at the board centre is shown next to the pose distance
as a sanity check.

    python charuco_demo.py                 # live, 1280x720 color
    python charuco_demo.py --depth         # live, plus depth cross-check
    python charuco_demo.py --image x.png   # detect on a still image (no camera)

Keys: q/ESC quit, s save snapshot to captures/.
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np

from board import add_board_args, board_from_args, make_detector


# --------------------------------------------------------------------------- detection

def detect_and_draw(image, board, detector, K=None, dist=None):
    """Detect the board, draw the overlay in place, return (n_corners, rvec, tvec)."""
    corners, ids, marker_corners, marker_ids = detector.detectBoard(image)

    if marker_ids is not None and len(marker_ids) > 0:
        cv2.aruco.drawDetectedMarkers(image, marker_corners, marker_ids)
    if ids is None or len(ids) == 0:
        return 0, None, None
    # OpenCV 5 returns flat (N,2)/(N,) arrays; the draw/match helpers expect (N,1,2)/(N,1).
    corners = np.asarray(corners, dtype=np.float32).reshape(-1, 1, 2)
    ids = np.asarray(ids, dtype=np.int32).reshape(-1, 1)
    cv2.aruco.drawDetectedCornersCharuco(image, corners, ids, (0, 0, 255))

    rvec = tvec = None
    # PnP needs >= 4 non-collinear points; 6 keeps the pose stable.
    if K is not None and len(ids) >= 6:
        obj_pts, img_pts = board.matchImagePoints(corners, ids)
        ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, dist, flags=cv2.SOLVEPNP_IPPE)
        if ok:
            axis_len = board.getSquareLength() * 2
            cv2.drawFrameAxes(image, K, dist, rvec, tvec, axis_len, 2)
        else:
            rvec = tvec = None
    return len(ids), rvec, tvec


def board_center(board):
    sx, sy = board.getChessboardSize()
    s = board.getSquareLength()
    return np.array([[sx * s / 2, sy * s / 2, 0.0]], dtype=np.float64)


def put_lines(image, lines, origin=(10, 30)):
    x, y = origin
    for line in lines:
        cv2.putText(image, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(image, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        y += 28


def pose_lines(n, rvec, tvec):
    lines = [f"charuco corners: {n}"]
    if tvec is not None:
        t = tvec.ravel() * 1000
        r = np.degrees(rvec.ravel())
        lines.append(f"t [mm]: x={t[0]:7.1f} y={t[1]:7.1f} z={t[2]:7.1f}")
        lines.append(f"rvec [deg]: {r[0]:6.1f} {r[1]:6.1f} {r[2]:6.1f}")
    return lines


# --------------------------------------------------------------------------- camera

def color_to_bgr(frame):
    from pyorbbecsdk import OBFormat

    w, h, fmt = frame.get_width(), frame.get_height(), frame.get_format()
    data = np.asanyarray(frame.get_data())
    if fmt == OBFormat.MJPG:
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    if fmt == OBFormat.RGB:
        return cv2.cvtColor(data.reshape(h, w, 3), cv2.COLOR_RGB2BGR)
    if fmt == OBFormat.BGR:
        return data.reshape(h, w, 3).copy()
    if fmt in (OBFormat.YUYV, OBFormat.YUY2):
        return cv2.cvtColor(data.reshape(h, w, 2), cv2.COLOR_YUV2BGR_YUYV)
    if fmt == OBFormat.NV12:
        return cv2.cvtColor(data.reshape(h * 3 // 2, w), cv2.COLOR_YUV2BGR_NV12)
    print(f"unsupported color format: {fmt}")
    return None


def pick_color_profile(pipeline, width, height, fps):
    from pyorbbecsdk import OBFormat, OBSensorType

    profiles = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
    for fmt in (OBFormat.RGB, OBFormat.MJPG):
        try:
            return profiles.get_video_stream_profile(width, height, fmt, fps)
        except Exception:
            pass
    print(f"{width}x{height}@{fps} not available, using the default color profile")
    return profiles.get_default_video_stream_profile()


def intrinsics_from_profile(profile):
    """Factory intrinsics -> OpenCV K and distortion (k1 k2 p1 p2 k3 k4 k5 k6)."""
    i = profile.get_intrinsic()
    d = profile.get_distortion()
    K = np.array([[i.fx, 0, i.cx], [0, i.fy, i.cy], [0, 0, 1]], dtype=np.float64)
    dist = np.array([d.k1, d.k2, d.p1, d.p2, d.k3, d.k4, d.k5, d.k6], dtype=np.float64)
    return K, dist


def run_camera(args, board, detector):
    from pyorbbecsdk import (AlignFilter, Config, Context, OBFrameAggregateOutputMode,
                             OBSensorType, OBStreamType, Pipeline)

    devices = Context().query_devices()
    if devices.get_count() == 0:
        sys.exit("No Orbbec device found. Check the USB-C cable (USB 3 port) and that the "
                 "udev rules are installed (scripts/install_udev_rules.sh).")
    info = devices.get_device_by_index(0).get_device_info()
    print(f"device: {info.get_name()}  serial: {info.get_serial_number()}  "
          f"fw: {info.get_firmware_version()}")

    pipeline = Pipeline()
    config = Config()
    color_profile = pick_color_profile(pipeline, args.width, args.height, args.fps)
    config.enable_stream(color_profile)
    print(f"color: {color_profile.get_width()}x{color_profile.get_height()} "
          f"{color_profile.get_format()} @ {color_profile.get_fps()} fps")

    align = None
    if args.depth:
        depth_profile = pipeline.get_stream_profile_list(
            OBSensorType.DEPTH_SENSOR).get_default_video_stream_profile()
        config.enable_stream(depth_profile)
        config.set_frame_aggregate_output_mode(OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE)
        align = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)
        print(f"depth: {depth_profile.get_width()}x{depth_profile.get_height()} "
              f"{depth_profile.get_format()} (aligned to color)")

    K, dist = intrinsics_from_profile(color_profile)
    print("K =\n", K, "\ndist =", dist)

    pipeline.start(config)
    win = "Femto Bolt ChArUco  |  q/ESC quit, s save"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    os.makedirs("captures", exist_ok=True)
    t_prev = time.time()
    try:
        while True:
            frames = pipeline.wait_for_frames(1000)
            if frames is None:
                continue
            if align is not None:
                frames = align.process(frames)
                if not frames:
                    continue
                frames = frames.as_frame_set()
            color_frame = frames.get_color_frame()
            if color_frame is None:
                continue
            image = color_to_bgr(color_frame)
            if image is None:
                continue
            raw = image.copy()

            n, rvec, tvec = detect_and_draw(image, board, detector, K, dist)
            lines = pose_lines(n, rvec, tvec)

            depth_frame = frames.get_depth_frame() if align is not None else None
            if depth_frame is not None and tvec is not None:
                lines.append(depth_check(depth_frame, board, K, dist, rvec, tvec))

            now = time.time()
            lines.append(f"{1.0 / max(now - t_prev, 1e-6):4.1f} fps")
            t_prev = now
            put_lines(image, lines)

            cv2.imshow(win, image)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("s"):
                stamp = time.strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(f"captures/{stamp}_raw.png", raw)
                cv2.imwrite(f"captures/{stamp}_overlay.png", image)
                print(f"saved captures/{stamp}_*.png")
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


def depth_check(depth_frame, board, K, dist, rvec, tvec):
    """Compare depth-sensor Z at the board centre with the ChArUco pose Z."""
    h, w = depth_frame.get_height(), depth_frame.get_width()
    depth_mm = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape(h, w)
    depth_mm = depth_mm.astype(np.float32) * depth_frame.get_depth_scale()

    centre_cam = cv2.Rodrigues(rvec)[0] @ board_center(board).T + tvec
    pose_z = centre_cam[2, 0] * 1000
    (u, v), = cv2.projectPoints(board_center(board), rvec, tvec, K, dist)[0].reshape(-1, 2)
    u, v = int(round(u)), int(round(v))
    patch = depth_mm[max(v - 5, 0):v + 6, max(u - 5, 0):u + 6]
    valid = patch[patch > 0]
    if valid.size == 0:
        return f"depth @ centre: n/a   pose z: {pose_z:.0f} mm"
    d = float(np.median(valid))
    return f"depth @ centre: {d:.0f} mm   pose z: {pose_z:.0f} mm   diff: {d - pose_z:+.0f} mm"


# --------------------------------------------------------------------------- still image

def run_image(args, board, detector):
    image = cv2.imread(args.image)
    if image is None:
        sys.exit(f"could not read {args.image}")
    n, _, _ = detect_and_draw(image, board, detector)
    print(f"{args.image}: {n} charuco corners detected")
    put_lines(image, [f"charuco corners: {n}"])
    out = os.path.splitext(args.image)[0] + "_detected.png"
    cv2.imwrite(out, image)
    print(f"wrote {out}")
    return n


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image", help="run on a still image instead of the camera")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--depth", action="store_true", help="also stream depth and compare with pose")
    add_board_args(p)
    args = p.parse_args()

    board = board_from_args(args)
    detector = make_detector(board)
    if args.image:
        run_image(args, board, detector)
    else:
        run_camera(args, board, detector)


if __name__ == "__main__":
    main()
