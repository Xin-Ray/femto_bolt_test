#!/bin/sh
# Install Orbbec udev rules (includes Femto Bolt, 2bc5:066b) and give users camera access.
#
#   sudo ./scripts/install_udev_rules.sh                     # default users below
#   sudo ./scripts/install_udev_rules.sh alice bob           # explicit users
#
# The rules are system-wide and set MODE 0666, so every account can open the
# camera. Users are also added to video/plugdev as a fallback; group changes
# take effect at their next login.

set -e

if [ "$(id -u)" -ne 0 ]; then
    echo "Please run with sudo: sudo $0 $*"
    exit 1
fi

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
RULES="$SCRIPT_DIR/../udev/99-obsensor-libusb.rules"
USERS=${*:-"iotlabg1robot iotlab3d"}

install -m 644 "$RULES" /etc/udev/rules.d/99-obsensor-libusb.rules
echo "installed /etc/udev/rules.d/99-obsensor-libusb.rules"

for u in $USERS; do
    if id "$u" >/dev/null 2>&1; then
        usermod -aG video,plugdev "$u"
        echo "added $u to video, plugdev"
    else
        echo "skipping $u: no such user"
    fi
done

udevadm control --reload-rules
udevadm trigger --subsystem-match=usb
echo "udev rules reloaded - unplug and replug the camera"
