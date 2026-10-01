"""Live ChArUco detection + board pose on an Orbbec Femto Bolt.

Uses the camera's factory color intrinsics, so no calibration step is needed
to get a metric board pose. With --depth, the board centre is mapped into the
depth image with the factory color->depth extrinsics and the measured depth is
shown next to the pose distance as a sanity check.

    python charuco_demo.py                 # live, largest color resolution
    python charuco_demo.py --width 1280 --height 720
    python charuco_demo.py --depth         # live, plus depth cross-check
    python charuco_demo.py --image x.png   # detect on a still image (no camera)

With --depth a second window shows the colorized depth image, with the
detected board outline and centre projected into it.

Keys: q/ESC quit, s save snapshot to captures/.

Speed: frames are fetched and MJPG-decoded on a background thread, the board
is detected on a copy at most DETECT_WIDTH wide (corners then refined on the
full-resolution image), and the overlay is drawn on a screen-sized copy.
"""

import argparse
import os
import sys
import threading
import time

import cv2
import numpy as np

from board import add_board_args, board_from_args, make_detector

DETECT_WIDTH = 1920         # detect on a downscaled copy above this width
VIEW_SIZE = (1600, 900)     # max size of the displayed image
DEPTH_VIS_MAX_MM = 5000     # depth colormap range: 0 (red) .. this (dark blue)


# --------------------------------------------------------------------------- detection

def detect(image, board, detector, K=None, dist=None):
    """Detect the board and estimate its pose.

    Returns a dict with marker_corners, marker_ids, corners, ids, rvec, tvec
    (missing items are None), all in full-resolution pixel coordinates.
    """
    s = min(1.0, DETECT_WIDTH / image.shape[1])
    small = image if s == 1.0 else cv2.resize(image, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    corners, ids, marker_corners, marker_ids = detector.detectBoard(small)
    result = dict(marker_corners=None, marker_ids=None, corners=None, ids=None, rvec=None, tvec=None)

    if marker_ids is not None and len(marker_ids) > 0:
        result["marker_corners"] = [np.asarray(c, dtype=np.float32) / s for c in marker_corners]
        result["marker_ids"] = marker_ids
    if ids is None or len(ids) == 0:
        return result

    # OpenCV 5 returns flat (N,2)/(N,) arrays; the draw/match helpers expect (N,1,2)/(N,1).
    corners = np.asarray(corners, dtype=np.float32).reshape(-1, 1, 2) / s
    ids = np.asarray(ids, dtype=np.int32).reshape(-1, 1)
    if s < 1.0:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        win = int(np.ceil(1 / s)) + 2
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
        cv2.cornerSubPix(gray, corners, (win, win), (-1, -1), criteria)
    result["corners"], result["ids"] = corners, ids

    # PnP needs >= 4 non-collinear points; 6 keeps the pose stable.
    if K is not None and len(ids) >= 6:
        obj_pts, img_pts = board.matchImagePoints(corners, ids)
        ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, dist, flags=cv2.SOLVEPNP_IPPE)
        if ok:
            result["rvec"], result["tvec"] = rvec, tvec
    return result


def draw(view, det, board, K=None, dist=None):
    """Draw a detect() result on view, which may be a scaled copy of the detected image."""
    s = view.shape[1] / det["image_width"]
    if det["marker_ids"] is not None:
        cv2.aruco.drawDetectedMarkers(view, [c * s for c in det["marker_corners"]], det["marker_ids"])
    if det["ids"] is not None:
        cv2.aruco.drawDetectedCornersCharuco(view, det["corners"] * s, det["ids"], (0, 0, 255))
    if det["rvec"] is not None:
        K_view = K.copy()
        K_view[:2] *= s
        cv2.drawFrameAxes(view, K_view, dist, det["rvec"], det["tvec"],
                          board.getSquareLength() * 2, max(view.shape[1] // 640, 2))


def n_corners(det):
    return 0 if det["ids"] is None else len(det["ids"])


def board_center(board):
    sx, sy = board.getChessboardSize()
    s = board.getSquareLength()
    return np.array([[sx * s / 2, sy * s / 2, 0.0]], dtype=np.float64)


def put_lines(image, lines, s=None):
    # Sized for 1280 px wide and scaled with the image (or by s), so text stays readable.
    if s is None:
        s = max(image.shape[1] / 1280, 0.5)
    x, y = int(10 * s), int(30 * s)
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.7 * s, max(int(2 * s), 1)
    pad = int(4 * s)
    for line in lines:
        # A box rather than a thick black outline: glyph spacing grows with thickness,
        # so an outline drawn underneath drifts away from the text along the line.
        (tw, th), base = cv2.getTextSize(line, font, scale, thick)
        cv2.rectangle(image, (x - pad, y - th - pad), (x + tw + pad, y + base + pad // 2),
                      (0, 0, 0), -1)
        cv2.putText(image, line, (x, y), font, scale, (0, 255, 0), thick, cv2.LINE_AA)
        y += int(28 * s) + pad


def pose_lines(n, rvec, tvec):
    lines = [f"charuco corners: {n}"]
    if tvec is not None:
        t = tvec.ravel() * 1000
        r = np.degrees(rvec.ravel())
        lines.append(f"t [mm]: x={t[0]:7.1f} y={t[1]:7.1f} z={t[2]:7.1f}")
        lines.append(f"rvec [deg]: {r[0]:6.1f} {r[1]:6.1f} {r[2]:6.1f}")
    return lines


# --------------------------------------------------------------------------- camera

def color_payload(frame):
    """Copy a color frame out of the SDK buffer: (format, width, height, data, host time in us)."""
    return (frame.get_format(), frame.get_width(), frame.get_height(),
            np.array(frame.get_data(), dtype=np.uint8, copy=True), frame.get_system_timestamp_us())


def color_to_bgr(frame):
    return decode_color(*color_payload(frame)[:4])


def decode_color(fmt, w, h, data):
    from pyorbbecsdk import OBFormat

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


def depth_to_mm(frame):
    h, w = frame.get_height(), frame.get_width()
    depth = np.frombuffer(frame.get_data(), dtype=np.uint16).reshape(h, w)
    return depth.astype(np.float32) * frame.get_depth_scale()


def pick_color_profile(pipeline, width, height, fps):
    """Color profile at width x height (0 x 0 = largest available at this fps).

    MJPG first: it's the camera's native color format, and decoding it here (on
    the grab thread) is faster than the SDK's own RGB conversion.
    """
    from pyorbbecsdk import OBFormat, OBSensorType

    profiles = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
    if width == 0 or height == 0:
        for fmt in (OBFormat.MJPG, OBFormat.RGB):
            matches = [profiles[i] for i in range(len(profiles))
                       if profiles[i].get_format() == fmt and profiles[i].get_fps() == fps]
            if matches:
                return max(matches, key=lambda p: p.get_width() * p.get_height())
        print(f"no MJPG/RGB profile at {fps} fps, using the default color profile")
        return profiles.get_default_video_stream_profile()
    for fmt in (OBFormat.MJPG, OBFormat.RGB):
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


class FrameGrabber:
    """Fetches frames on one thread and decodes color on DECODE_THREADS others.

    Only the newest frame is ever kept: the fetch thread drains the SDK queue,
    and decoders always take the newest undecoded frame. A 4K MJPG decode takes
    ~35 ms, so with one decoder any extra load drops below 30 fps; two give headroom
    without adding latency. Color and depth are kept separately, so color runs at
    its own rate whatever depth does.
    """

    DECODE_THREADS = 2

    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.cond = threading.Condition()
        self.running = True
        self.raw, self.raw_seq, self.taken_seq = None, 0, 0     # newest undecoded color
        self.color, self.color_ts, self.color_seq, self.shown_seq = None, 0, 0, 0
        self.depth = None
        self.threads = [threading.Thread(target=self._fetch, daemon=True)]
        self.threads += [threading.Thread(target=self._decode, daemon=True)
                         for _ in range(self.DECODE_THREADS)]

    def start(self):
        for t in self.threads:
            t.start()

    def _fetch(self):
        while self.running:
            frames = self.pipeline.wait_for_frames(100)
            if frames is None:
                continue
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            # Drain whatever queued up meanwhile and keep only the newest of each;
            # processing stale frames is what builds up latency.
            while True:
                frames = self.pipeline.wait_for_frames(0)
                if frames is None:
                    break
                color_frame = frames.get_color_frame() or color_frame
                depth_frame = frames.get_depth_frame() or depth_frame
            depth = depth_to_mm(depth_frame) if depth_frame is not None else None
            raw = color_payload(color_frame) if color_frame is not None else None
            with self.cond:
                if depth is not None:
                    self.depth = depth
                if raw is not None:
                    self.raw, self.raw_seq = raw, self.raw_seq + 1
                    self.cond.notify_all()

    def _decode(self):
        while self.running:
            with self.cond:
                if not self.cond.wait_for(lambda: self.raw_seq > self.taken_seq or not self.running,
                                          timeout=0.1) or not self.running:
                    continue
                seq, raw, self.taken_seq = self.raw_seq, self.raw, self.raw_seq
            image = decode_color(*raw[:4])
            with self.cond:
                if image is not None and seq > self.color_seq:
                    self.color, self.color_ts, self.color_seq = image, raw[4], seq
                    self.cond.notify_all()

    def latest(self, timeout=1.0):
        """Newest (color, depth, color host timestamp in us) once a new color frame is
        decoded, else (None, None, None)."""
        with self.cond:
            if not self.cond.wait_for(lambda: self.color_seq > self.shown_seq, timeout):
                return None, None, None
            self.shown_seq = self.color_seq
            return self.color, self.depth, self.color_ts

    def stop(self):
        self.running = False
        with self.cond:
            self.cond.notify_all()
        for t in self.threads:
            t.join(timeout=2)


def run_camera(args, board, detector):
    from pyorbbecsdk import Config, Context, OBFrameAggregateOutputMode, OBSensorType, Pipeline

    # Keep the Context alive: the device list and devices reference its device manager.
    ctx = Context()
    devices = ctx.query_devices()
    if devices.get_count() == 0:
        sys.exit("No Orbbec device found. Check the USB-C cable (USB 3 port) and that the "
                 "udev rules are installed (scripts/install_udev_rules.sh).")
    device = devices.get_device_by_index(0)
    info = device.get_device_info()
    print(f"device: {info.get_name()}  serial: {info.get_serial_number()}  "
          f"fw: {info.get_firmware_version()}  usb: {info.get_connection_type()}")
    if info.get_connection_type().startswith("USB2"):
        print("warning: connected over USB 2 - use a USB 3 port/cable for full depth frame rate")

    pipeline = Pipeline(device)
    config = Config()
    color_profile = pick_color_profile(pipeline, args.width, args.height, args.fps)
    config.enable_stream(color_profile)
    print(f"color: {color_profile.get_width()}x{color_profile.get_height()} "
          f"{color_profile.get_format()} @ {color_profile.get_fps()} fps")
    K, dist = intrinsics_from_profile(color_profile)
    print("K =\n", K, "\ndist =", dist)

    depth_calib = None
    if args.depth:
        depth_profile = pipeline.get_stream_profile_list(
            OBSensorType.DEPTH_SENSOR).get_default_video_stream_profile()
        config.enable_stream(depth_profile)
        # Deliver each frame as soon as it arrives. Any aggregation mode makes the SDK
        # hold color back (~0.5 s over USB 2) while it waits for a matching depth frame.
        config.set_frame_aggregate_output_mode(OBFrameAggregateOutputMode.DISABLE)
        ext = color_profile.get_extrinsic_to(depth_profile)
        depth_calib = dict(
            R=np.array(ext.rot, dtype=np.float64).reshape(3, 3),
            t=np.array(ext.transform, dtype=np.float64).reshape(3, 1),
            **dict(zip(("K", "dist"), intrinsics_from_profile(depth_profile))))
        print(f"depth: {depth_profile.get_width()}x{depth_profile.get_height()} "
              f"{depth_profile.get_format()} @ {depth_profile.get_fps()} fps")

    w, h = color_profile.get_width(), color_profile.get_height()
    view_scale = min(1.0, VIEW_SIZE[0] / w, VIEW_SIZE[1] / h)

    pipeline.start(config)
    grabber = FrameGrabber(pipeline)
    grabber.start()
    win = "Femto Bolt ChArUco  |  q/ESC quit, s save"
    # Fixed-size windows that match the images exactly (they are already resized to
    # VIEW_SIZE). A resizable window with OpenCV's Qt toolbar ends up slightly
    # smaller than the image and gets software-rescaled on every frame (~10 ms).
    win_flags = cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL
    cv2.namedWindow(win, win_flags)
    depth_win = "Femto Bolt depth"
    if args.depth:
        cv2.namedWindow(depth_win, win_flags)
        dw, dh = depth_profile.get_width(), depth_profile.get_height()
        depth_view_size = (round(dw * h * view_scale / dh), round(h * view_scale))
    os.makedirs("captures", exist_ok=True)
    t_prev = time.time()
    fps = 0.0
    try:
        while True:
            image, depth_mm, stamp_us = grabber.latest()
            if image is None:
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
                continue

            det = detect(image, board, detector, K, dist)
            det["image_width"] = image.shape[1]
            view = image if view_scale == 1.0 else cv2.resize(
                image, None, fx=view_scale, fy=view_scale, interpolation=cv2.INTER_AREA)
            draw(view, det, board, K, dist)
            lines = pose_lines(n_corners(det), det["rvec"], det["tvec"])
            depth_view = None
            if depth_mm is not None:
                depth_view = cv2.resize(colorize_depth(depth_mm), depth_view_size,
                                        interpolation=cv2.INTER_NEAREST)
                if det["tvec"] is not None:
                    text = depth_check(depth_mm, depth_view, depth_calib, board,
                                       det["rvec"], det["tvec"])
                    lines.append(text)
                    put_lines(depth_view, text.split("   "))

            now = time.time()
            fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-6)
            t_prev = now
            # Time since the frame reached the host (excludes exposure and USB transfer).
            lines.append(f"{fps:4.1f} fps   latency {now * 1000 - stamp_us / 1000:3.0f} ms")
            put_lines(view, lines)

            cv2.imshow(win, view)
            if depth_view is not None:
                cv2.imshow(depth_win, depth_view)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("s"):
                stamp = time.strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(f"captures/{stamp}_raw.png", image)
                cv2.imwrite(f"captures/{stamp}_overlay.png", view)
                if depth_mm is not None:
                    cv2.imwrite(f"captures/{stamp}_depth_mm.png", depth_mm.astype(np.uint16))
                    cv2.imwrite(f"captures/{stamp}_depth_view.png", depth_view)
                print(f"saved captures/{stamp}_*.png")
    finally:
        grabber.stop()
        pipeline.stop()
        cv2.destroyAllWindows()


def colorize_depth(depth_mm):
    """Depth in mm -> BGR image (near red, far blue, no data black)."""
    # Saturates beyond DEPTH_VIS_MAX_MM; inverted so near is red.
    scaled = cv2.bitwise_not(cv2.convertScaleAbs(depth_mm, alpha=255.0 / DEPTH_VIS_MAX_MM))
    vis = cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)
    return cv2.bitwise_and(vis, vis, mask=cv2.compare(depth_mm, 0, cv2.CMP_GT))


def board_to_depth_pixels(points_board, calib, rvec, tvec):
    """Board-frame points (m) -> (depth camera points in mm, depth image pixels)."""
    pts_color = (cv2.Rodrigues(rvec)[0] @ points_board.T + tvec) * 1000
    pts_depth = calib["R"] @ pts_color + calib["t"]
    px = cv2.projectPoints(pts_depth.T, np.zeros(3), np.zeros(3),
                           calib["K"], calib["dist"])[0].reshape(-1, 2)
    return pts_depth, px


def depth_check(depth_mm, depth_view, calib, board, rvec, tvec):
    """Compare depth-sensor Z at the board centre with the ChArUco pose.

    The board (from the color pose) is moved into the depth camera frame with
    the factory extrinsics and projected into the raw depth image, so no
    full-frame alignment is needed. Its outline and centre are drawn on depth_view,
    which may be a resized copy of the depth image.
    """
    s = depth_view.shape[1] / depth_mm.shape[1]
    sx, sy = board.getChessboardSize()
    sq = board.getSquareLength()
    outline = np.array([[0, 0, 0], [sx * sq, 0, 0], [sx * sq, sy * sq, 0], [0, sy * sq, 0]])
    _, outline_px = board_to_depth_pixels(outline, calib, rvec, tvec)
    cv2.polylines(depth_view, [np.round(outline_px * s).astype(np.int32)], True, (255, 255, 255), 2)

    centre_depth, ((u, v),) = board_to_depth_pixels(board_center(board), calib, rvec, tvec)
    pose_z = centre_depth[2, 0]
    u, v = int(round(u)), int(round(v))
    cv2.drawMarker(depth_view, (round(u * s), round(v * s)), (255, 255, 255), cv2.MARKER_CROSS, 20, 2)
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
    det = detect(image, board, detector)
    det["image_width"] = image.shape[1]
    draw(image, det, board)
    n = n_corners(det)
    print(f"{args.image}: {n} charuco corners detected")
    put_lines(image, [f"charuco corners: {n}"])
    out = os.path.splitext(args.image)[0] + "_detected.png"
    cv2.imwrite(out, image)
    print(f"wrote {out}")
    return n


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image", help="run on a still image instead of the camera")
    p.add_argument("--width", type=int, default=0, help="color width (default 0 = largest available)")
    p.add_argument("--height", type=int, default=0, help="color height (default 0 = largest available)")
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
