# MD1000 NPU INT8 — 排查记录与交接文档

> 脱敏版：所有主机地址与凭据已替换为占位符。

## 环境与硬件

| 项目 | 值 |
|---|---|
| 开发板 | 浪潮 MD1000（RK3566，单核 NPU ~1TOPS） |
| SSH | `user@<BOARD_IP>`（sudo 需密码） |
| 系统 | Armbian 26.08 trixie，内核 `6.18.39-ophub` |
| DTB | `rockchip/rk3566-md1000.dtb`（带 `__symbols__`，有 scmi/cru/power/qos_npu 符号，无 vdd_npu 供电轨） |
| WSL | Ubuntu（用户目录），conda 环境 `toolkit2_1.6` 内 rknn-toolkit2 2.3.2（官方 GitHub wheel）+ `venv_tk16`（rknn-toolkit2 1.6.0） |
| 校准数据 | COCO128（128 张 COCO 图） |

## 核心问题与根因：INT8 板上执行损坏

### 现象（确定性、可复现）

- 同一模型同一图片：x86 模拟器输出完全正常；板上 NPU int8 输出系统性损坏（cls 饱和、box 均值 -1.75）
- 全局性：yolov8n、yolov5s_relu 的 int8 均损坏（fp16 vs int8 相关性 -0.22）；FP16 一直正常

### 已排除的原因（全部实测）

| # | 假设 | 实验 | 结果 |
|---|---|---|---|
| 1 | 量化算法差 | normal / kl_divergence | 都坏 |
| 2 | 校准图颜色顺序 | RGB2BGR False/True | 都坏 |
| 3 | 输出反量化 | output_optimize False | 都坏 |
| 4 | 多核调度元数据 | single_core_mode | 都坏 |
| 5 | toolkit 构建差异 | 官方 wheel 重装 | 都坏 |
| 6 | librknnrt 被裁剪 | md5 与官方一致 | 排除 |
| 7 | codex 驱动改动 | pristine 源码重建 | 一模一样地坏 |
| 8 | NO_GKI 缺失 | 加 `-DCONFIG_NO_GKI=1` 重编译 | 都坏 |
| 9 | IOMMU 模式 | 启用 iommu overlay | 更糟（fp16 也坏，任务提交失败——iommu glue 未适配主线 iommu），已回退 |

### 决定性发现

**官方 1.x 工具链编译的 mobilenet_v1 int8（rknpu2 仓库自带 RK3566 模型）在板上完全正确**（top-1 score=0.9844，softmax 分布健康）→ int8 硬件执行正常！

**根因**：2.3.2 工具链生成的 int8 模型任务链在移植驱动上断链：
- verbose 捕获：模型 758 个任务，`failed to submit ... run task counter: 184/758`
- FP16 任务链短 → 正常；1.x 格式 int8 任务链简单 → 正常；2.x 格式 int8 长任务链 → 必坏

**修复路径**：
1. ✅ 绕过（已实施）：toolkit2 1.6.0 重转（1.x 模型格式）
2. 根治（待做）：调试移植驱动 `rknpu_job.c` 任务链逻辑（fence/多 job 提交对长链的处理）；调试接口 `/sys/kernel/debug/rknpu/`

## 其他确认事项

- 板子反复"失联"为**突然掉电**（journal 戛然而止、无 panic、温度 31°C、内存 12% —— 排除软件崩溃/OOM/过热），更换电源适配器可解决
- NPU 时钟被固件 SCMI 锁 594-600MHz（800MHz OPP 不可达）
- 运行时查不到 NPU 频率（`Failed to get NPU freq, take 900MHz`）——驱动移植未实现频率查询接口，只影响统计显示
- 板子双网卡：有线口稳定、无线口丢包严重，建议固定使用有线
- tk16（1.6.0 转换）模型的 sum 分支输出恒为 1.0（ConvClip 转换饱和），实时后处理的预过滤需改用 cls max 实现
- USB 3.0 口实测：Kingston DT 100 G3 协商 5000 Mbps，读 82 MB/s、写 10 MB/s（盘本身瓶颈）
- USB 相机：Generic USB Camera（USB 2.0），YUYV 320×240@30 直采免缩放

## 性能优化记录（相机实时检测）

| 版本 | 端到端 FPS | 关键改动 |
|---|---|---|
| v1（416 串行） | 13.9 | 采集 33ms + 推理 38ms 串行阻塞 |
| v2（320 流水线） | 20.1 | 320 输入模型 + 采集/推理双线程 |
| v3 | 21.7 | letterbox 移入采集线程 |
| v4（最终） | **29.9** | YUYV 320×240 直采免缩放 + cls 置信度预过滤（DFL 只算候选格）+ 去冗余拷贝 |

已知坑：
- 1.6.0 量化对 NHWC 输入 onnx 报错，需 NCHW 导出（fork 的 Detect rknn 优化分支要扩到 onnx 格式）
- default.yaml `batch: 16` 会导出固定 batch=16 的 onnx，量化器报 batch 错，需改 1
- 1.6.0 需要 onnx==1.14.1 / onnxoptimizer==0.2.7（requirements_cp38-1.6.0.txt）
- 后处理 K==0 早退分支返回值约定必须一致（空 box 4ch + None），否则空画面崩溃
