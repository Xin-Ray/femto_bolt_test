#!/bin/sh
# Run the ChArUco demo with the femto_bolt conda env's Python (no activation needed).
#
#   ./run_demo.sh                   # live, 1280x720 color
#   ./run_demo.sh --depth           # live, plus depth cross-check
#   ./run_demo.sh --image x.png     # still image, no camera
#
# Arguments are passed to charuco_demo.py. Override the interpreter with
# PYTHON=/path/to/python or the env name with FEMTO_ENV=name.

set -e

cd "$(dirname "$0")"

if [ -z "$PYTHON" ]; then
    ENV_NAME=${FEMTO_ENV:-femto_bolt}
    BASE=$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")
    PYTHON="$BASE/envs/$ENV_NAME/bin/python"
fi

if [ ! -x "$PYTHON" ]; then
    echo "No Python at $PYTHON"
    echo "Create the env (see README) or set PYTHON=/path/to/python"
    exit 1
fi

exec "$PYTHON" charuco_demo.py "$@"
