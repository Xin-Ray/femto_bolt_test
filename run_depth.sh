#!/bin/sh
# Run the ChArUco demo with depth enabled (depth at the board centre vs pose distance).
# Over USB 2 the depth value updates at ~7 fps; use USB 3 for the full 30 fps.
#
#   ./run_depth.sh
#   ./run_depth.sh --width 1920 --height 1080
#
# Arguments are passed to charuco_demo.py, same as run_demo.sh.

exec "$(dirname "$0")/run_demo.sh" --depth "$@"
