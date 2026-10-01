#!/bin/sh
# Run the ChArUco demo with the femto_bolt conda env's Python (no activation needed).
#
#   ./run_demo.sh                   # live, largest color resolution (3840x2160)
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

# opencv-python's Qt looks for fonts only in cv2/qt/fonts, which the wheel doesn't
# ship, and warns on every window. Point it at the system DejaVu fonts once.
FONTS=/usr/share/fonts/truetype/dejavu
for qt in "$(dirname "$PYTHON")"/../lib/python3*/site-packages/cv2/qt; do
    if [ -d "$qt" ] && [ ! -e "$qt/fonts" ] && [ -d "$FONTS" ]; then
        ln -s "$FONTS" "$qt/fonts" 2>/dev/null || true
    fi
done

exec "$PYTHON" charuco_demo.py "$@"
