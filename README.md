# MD1000 (RK3566) NPU — RKNPU 驱动移植 + YOLOv8 INT8 部署

浪潮 Inspur MD1000 开发板（RK3566）上完成 Rockchip RKNPU 内核驱动移植（接手自 codex 的 out-of-tree 移植工程），并打通 YOLOv8 INT8 全流程：量化转换 → 板上部署 → 检出质量验证 → USB 相机实时检测。

## 最终成绩

| 配置 | 延迟 | 吞吐 | 检出（COCO bus.jpg：3 person + 1 bus） |
|---|---|---|---|
| FP16 @640（起点） | 346 ms | 2.89 FPS | person 0.89/0.87/0.65 + bus 0.86 |
| INT8 @640（toolkit2 1.6.0） | 96 ms | 10.4 FPS | person 0.88/0.85/0.57 + bus 0.83 |
| INT8 @416 单流 | 34.4 ms | 29.08 FPS | person 0.83/0.65/0.34 + bus 0.81 |
| INT8 @416 ×2 客户端 | 38 ms | **52.3 FPS 聚合** | — |
| **USB 相机实时检测 @320** | 33.5 ms | **29.9 FPS**（相机 30fps 上限） | 实时流水线 |

## 关键经验

1. **INT8 在 NPU 上输出损坏的根因**：rknn-toolkit2 2.3.2 生成的 int8 模型任务链（~758 tasks）在本移植驱动上断链（debugfs 可见 `run task counter: 184/758`）；FP16 任务链正常、1.x 格式 int8 模型正常。
2. **解法**：用 **toolkit2 1.6.0**（1.x 模型格式）转换即完全正常，检出质量几乎无损（score 差 0.02~0.08）。1.6.0 官方 wheel 在 [airockchip/rknn-toolkit2 v1.6.0 tag](https://github.com/airockchip/rknn-toolkit2/tree/v1.6.0/rknn-toolkit2/packages)。
3. **416/320 输入重导出**：用 [airockchip/ultralytics_yolov8](https://github.com/airockchip/ultralytics_yolov8) fork（RKOPT 优化图），把 `Detect` 头的 rknn 分支条件扩到 `onnx`、`imgsz`/`batch` 改配置即可导出 RKNN 优化 onnx，再走 1.6.0 量化。
4. **相机实时**：YUYV 320×240 直采免缩放 + 采集/推理双线程流水线 + cls 置信度预过滤后处理（DFL 只算候选格），端到端打满相机 30fps。
5. IOMMU 模式在主线内核移植上不可用（`rknpu_iommu.c` 被 `CONFIG_NO_GKI` 门控、且主线 iommu glue 未适配），non-IOMMU + DRM GEM 是当前可用路径。

## 目录结构

```
├── driver/rknpu-md1000-sandbox/   # RKNPU v0.9.8 out-of-tree 内核模块（MD1000 移植版）
│   ├── src/                        # 驱动源码（含 vdd-supply 容忍补丁）
│   ├── dts/rk3566-md1000-rknpu-overlay.dts  # MD1000 设备树 overlay（SCMI 时钟/无供电轨/reg=0x100）
│   ├── Kbuild / Makefile / dkms.conf
├── conversion/                    # WSL/PC 侧模型转换
│   ├── quantize.py                 # 2.3.2 量化（用于对照/FP16）
│   ├── quantize16*.py              # 1.6.0 量化（INT8 正确产出的关键）
│   ├── export_onnx_*.sh            # ultralytics fork 导出 RKNN 优化 onnx（416/320）
├── board/                         # 板端推理/测试脚本（RKNNLite + numpy 后处理，免 torch）
│   ├── cam_detect_320.py           # USB 相机实时检测（流水线 + 预过滤）
│   ├── npu_yolov8_test*.py         # 单图检出质量测试
│   ├── bench_fps*.py / bench_client*.py  # 吞吐基准（单/多客户端）
├── docs/PROGRESS.md               # 完整排查记录与交接文档
```

## 环境要求

- 板端：Armbian（主线内核 6.x + linux-headers）、rknn-toolkit-lite2 2.3.2、官方 `librknnrt.so` 2.3.2、opencv-python
- 转换端：Python 3.8 + rknn-toolkit2 **1.6.0**（INT8 正确产出）；2.3.2 仅用于模拟器对照

## 快速上手

```bash
# 板端编译驱动（需 linux-headers）
cd driver/rknpu-md1000-sandbox && make && sudo make install && sudo depmod -a
# overlay 编译安装（cpp + dtc + fdtoverlay，见 conversion/../board/deploy_overlay.sh）

# 转换 INT8（PC/WSL，1.6.0 环境）
python conversion/quantize16_320.py

# 板端相机实时检测
python board/cam_detect_320.py
```

## 致谢与参考

- 驱动移植基础：[w568w/rknpu-module](https://github.com/w568w/rknpu-module)（基于 rockchip-linux develop-6.6 BSP 的 out-of-tree 移植）
- [airockchip/rknn-toolkit2](https://github.com/airockchip/rknn-toolkit2)、[airockchip/rknpu2](https://github.com/airockchip/rknpu2)、[airockchip/rknn-model-zoo](https://github.com/airockchip/rknn-model-zoo)、[airockchip/ultralytics_yolov8](https://github.com/airockchip/ultralytics_yolov8)
