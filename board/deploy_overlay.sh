#!/bin/bash
set -x
SANDBOX=~/rknn-yolov8/driver-audit/rknpu-md1000-sandbox
KROOT=/usr/src/linux-headers-$(uname -r)
cd $SANDBOX/dts
cpp -nostdinc -I $KROOT/include -undef -x assembler-with-cpp rk3566-md1000-rknpu-overlay.dts rk3566-md1000-rknpu-overlay.dts.preprocessed
dtc -@ -I dts -O dtb -o rk3566-md1000-rknpu.dtbo rk3566-md1000-rknpu-overlay.dts.preprocessed 2>&1 | grep -c 'Error'
rm -f rk3566-md1000-rknpu-overlay.dts.preprocessed
echo "$SUDO_PASS" | sudo -S install -D -m 644 rk3566-md1000-rknpu.dtbo /boot/overlay-user/rk3566-md1000-rknpu.dtbo
fdtoverlay -i /boot/dtb/rockchip/rk3566-md1000.dtb -o /tmp/test.dtb rk3566-md1000-rknpu.dtbo && echo APPLY_OK
dtc -I dtb -O dts /tmp/test.dtb 2>/dev/null | grep -A3 'npu@fde40000' | grep reg
sleep 2
echo "$SUDO_PASS" | sudo -S reboot
