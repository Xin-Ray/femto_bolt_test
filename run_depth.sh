#!/bin/sh
# Run the ChArUco demo with depth enabled (aligned to color, depth-vs-pose check).
# Needs a USB 3 connection for a usable frame rate.
#
#   ./run_depth.sh
#   ./run_depth.sh --width 1920 --height 1080
#
# Arguments are passed to charuco_demo.py, same as run_demo.sh.

exec "$(dirname "$0")/run_demo.sh" --depth "$@"
