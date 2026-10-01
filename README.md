# Femto Bolt ChArUco demo

Live ChArUco board detection and 6-DoF board pose on an Orbbec Femto Bolt,
using the camera's factory color intrinsics (no calibration step needed).

## Board

`boards/charuco_a4_7x5_38mm_4x4.pdf` is a 7x5-square board with 38 mm squares,
28 mm markers and `DICT_4X4_50`. Print it at **100% / actual size** on A4, then
check that 7 squares measure 266 mm. If they don't, pass the measured size with
`--square` and `--marker` (in metres). Defaults are set in `board.py`.

## Setup

1. Install the udev rules once (system-wide; covers all users):

   ```bash
   sudo ./scripts/install_udev_rules.sh            # iotlabg1robot + iotlab3d
   ```

   Then unplug and replug the camera. Plug it into a **USB 3** port with the
   USB-C data cable; the Femto Bolt also needs its 12 V power supply unless
   the port supplies enough power.

2. Python environment (already created on this machine as conda env `femto_bolt`):

   ```bash
   conda create -n femto_bolt python=3.12
   conda activate femto_bolt
   pip install -r requirements.txt
   ```

   Install only `opencv-python`, not `opencv-contrib-python` as well. Both
   provide `cv2` and conflict. ArUco/ChArUco is in the main package.

## Run

```bash
conda activate femto_bolt
python charuco_demo.py                  # 1280x720 color, pose overlay
python charuco_demo.py --depth          # + depth-vs-pose distance check at board centre
python charuco_demo.py --width 1920 --height 1080
python charuco_demo.py --image photo.png    # offline, no camera
```

Keys: `q`/`Esc` quit, `s` saves raw and overlay frames to `captures/`.

The overlay shows detected markers, the interpolated ChArUco corners (24 when the
whole board is visible) and the board frame axes (origin at the top-left
chessboard corner, X red, Y green, Z blue). The translation is the board
origin in the color camera frame, in mm.

## Notes

- Driver: Orbbec SDK v2 via the `pyorbbecsdk2` wheel. It bundles `libOrbbecSDK.so`,
  so nothing else has to be built. The bundled examples are in
  `$CONDA_PREFIX/lib/python3.12/site-packages/pyorbbecsdk/examples/`.
- The SDK's camera sanity check: `python $CONDA_PREFIX/lib/python3.12/site-packages/pyorbbecsdk/examples/quick_start.py`
