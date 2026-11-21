# MD1000 驱动和设备树

适用环境为浪潮 MD1000（RK3566）、Armbian 和 `6.18.39-ophub`。下表列出这块板上启用的配置。

| 设备 | 驱动与设备树 | 实板验证 |
| --- | --- | --- |
| NPU 调频和调压 | RKNPU 接入 Linux devfreq/OPP，使用 RK809 DCDC_REG4 | 200／400 MHz 为 875 mV，600 MHz 为 900 mV |
| NPU IOMMU | 设备树启用 `iommu@fde4b000`，GEM 缓存同步按物理散列表的 `orig_nents` 遍历 | IOMMU 模式通过 INT8 推理回归 |
| VDPU346 | `rockchip-vdec` 回移补丁，overlay 设置 RK356x 的寄存器窗口与解码时钟 | GStreamer 1.28 的 V4L2 stateless H.264／HEVC 解码接入 YOLOv8 |
| GPU 调压 | Panfrost 使用 RK809 DCDC_REG2 | 200／300／400 MHz 档为 875 mV，600 MHz 档为 900 mV，渲染输出一致 |

800 MHz 在单板测试的临时时钟配置下跑完了约 32 分钟，随后恢复到 600 MHz。发布配置中启用的最高档为 600 MHz。

## 构建 RKNPU 和 overlay

在 MD1000 上使用与运行内核匹配的头文件、编译器和 `device-tree-compiler`。

```bash
make -C rknpu-md1000-sandbox all md1000-dtbo

dts=rknpu-md1000-sandbox/dts
fdtoverlay -i /boot/dtb/rockchip/rk3566-md1000.dtb -o md1000-check.dtb \
  "$dts/rk3566-md1000-rknpu.dtbo" \
  "$dts/rk3566-md1000-vdpu346.dtbo" \
  "$dts/rk3566-md1000-npu-dvfs.dtbo" \
  "$dts/rk3566-md1000-npu-iommu-fix.dtbo" \
  "$dts/rk3566-md1000-gpu-dvfs.dtbo"
```

## 构建 VDPU346

回移补丁的源码基线为 ophub 的 `97311de7b046ee6bc8f89bb3cd0815cabd079757`。该基线的 `rkvdec` 源码应用补丁后与实板构建所用源码一致。构建使用实板 `6.18.39-ophub` 的头文件与符号表。补丁来自 Daniel Golle 和 Christian Hewitt 的 VDPU346 系列及 HEVC RPS 适配。[源码基线](https://github.com/ophub/linux-6.18.y/tree/97311de7b046ee6bc8f89bb3cd0815cabd079757/drivers/media/platform/rockchip/rkvdec) · [VDPU346 补丁系列](https://lore.kernel.org/linux-media/cover.1794654074.git.daniel@makrotopia.org/)

在本仓库根目录运行，下例使用单独的 Linux 源码目录。

```bash
project=$PWD
git init linux-md1000-vdec
git -C linux-md1000-vdec remote add origin https://github.com/ophub/linux-6.18.y.git
git -C linux-md1000-vdec fetch --depth=1 --filter=blob:none origin 97311de7b046ee6bc8f89bb3cd0815cabd079757
git -C linux-md1000-vdec sparse-checkout set drivers/media/platform/rockchip/rkvdec
git -C linux-md1000-vdec checkout --detach FETCH_HEAD
git -C linux-md1000-vdec apply --check "$project/patches/vdpu346-6.18.39-backport.patch"
git -C linux-md1000-vdec apply "$project/patches/vdpu346-6.18.39-backport.patch"

make -C "/lib/modules/$(uname -r)/build" \
  M="$project/linux-md1000-vdec/drivers/media/platform/rockchip/rkvdec" \
  CONFIG_VIDEO_ROCKCHIP_VDEC=m modules
```

## 启动配置和复测

部署前备份 `/boot/armbianEnv.txt` 和要替换的旧文件。把编译得到的 `rknpu.ko`、`rockchip-vdec.ko` 放在当前内核的 `/lib/modules/<版本>/updates/`，运行 `sudo depmod -a`。把上面通过合并检查的五个 `.dtbo` 放入 `/boot/overlay-user/`。在 `/boot/armbianEnv.txt` 里按实板的顺序启用，原有其他配置按需保留。

```ini
user_overlays=rk3566-md1000-rknpu rk3566-md1000-vdpu346 rk3566-md1000-npu-dvfs rk3566-md1000-npu-iommu-fix rk3566-md1000-gpu-dvfs
```

重启后检查加载的模块、NPU 工作点和 IOMMU 组。如果启用失败，恢复之前的启动配置和旧模块，重新运行 `sudo depmod -a` 后重启。

```bash
modinfo -n rknpu
modinfo -n rockchip_vdec
cat /sys/class/devfreq/fde40000.npu/available_frequencies
readlink /sys/bus/platform/devices/fde40000.npu/iommu_group
gst-inspect-1.0 v4l2slh264dec
gst-inspect-1.0 v4l2slh265dec
```

预期两个模块均来自 `updates/`，NPU 列出 `200000000 400000000 600000000` 并链接到 IOMMU 组。若安装了支持 V4L2 stateless 的 GStreamer，可用 H.264／HEVC 文件分别验证解码。实板的 NPU、GPU 和 VDEC 电源域在负载结束后均恢复关闭。
