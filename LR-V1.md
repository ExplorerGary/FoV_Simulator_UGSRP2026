# LR-V1：500 ms Progressive FoV Streaming Predictor

## 1. 版本定义

`LR-V1` 是当前冻结的高质量 progressive FoV streaming operating point：

```text
过去 500 ms 的 Head 6DoF 与 gaze
        ↓
204 维固定特征
        ↓
多输出 Ridge Linear Regression
        ↓
500 ms 后每个 cell 的 visibility score
        ↓
threshold = 0.20 + one-step guard6
        ↓
每个 cell 选择 Base-only 或 Base + E3
```

这里的 `LR-V1` 是归档/发布版本名。它对应代码中的 `raw_gaze` feature mode（历史内部称
V3），不要与早期 head-only `raw_history` baseline（历史内部称 V1）混淆。

## 2. 输入

模型推理只读取当前时刻及之前的数据，不读取当前或未来的 cell visibility。

- 采样率：30 FPS。
- 历史窗口：500 ms，包含当前帧在内共 16 帧。
- Head：每帧 `LocationX/Y/Z` 与 `RotationRoll/Pitch/Yaw`，即 6DoF。
- Gaze：每帧 world-space unit gaze direction，即 `GazeHitX/Y/Z`。
- 仅使用完整 500 ms gaze history 有效的窗口。

固定特征共 204 维：

| Feature group | Dimensions |
|---|---:|
| 16 帧 raw Head 6DoF history | 96 |
| 当前 world-space gaze | 3 |
| 前 15 帧 gaze 相对当前 gaze 的差 | 45 |
| gaze 长期与短期速度 | 6 |
| gaze 加速度 | 3 |
| 归一化的 500 ms gaze 外推方向 | 3 |
| 16 帧 gaze 在 head forward/right/up basis 中的坐标 | 48 |
| 总计 | 204 |

## 3. 输出与监督目标

模型是一个多输出 Ridge Linear Regression（`alpha=1.0`）。它一步直接输出 500 ms 后
每个 cell 的连续 visibility score，不先预测未来 pose，也不调用几何可见性算法。

原始监督信号是 `contributing_gaussian_fraction ∈ [0,1]`；二值训练目标为：

```text
fraction >= 0.5  -> 1，未来可见，应当接收 E3
fraction <  0.5  -> 0，未来不可见
```

Feature 和每个 cell 的 target 分别使用训练数据统计量标准化；推理后将 score 恢复到原始
尺度并裁剪至 `[0,1]`。

## 4. 窗口与数据划分

- History window：500 ms。
- Prediction horizon：500 ms。
- 10 条完整 trace 按 8:2 划分，避免相邻帧跨训练集和测试集泄漏。
- 测试 trace：`26_7_29_12_37_21`、`26_7_31_15_1_21`。
- 最终评测：两条测试 trace 中 gaze history 完整有效的 3401 帧。

## 5. Threshold

最终 decision threshold 固定为 `0.20`：

```text
score >= 0.20 -> Base + E3
score <  0.20 -> Base-only
```

该 operating point 有意偏向 recall，因为漏传未来真正可见的 cell 会造成空洞或显著视觉
伪影，其代价通常高于多传输一些暂时不可见的 cell。

## 6. 特殊操作：guard6

`guard6` 是 streaming policy 的空间安全边界，不是另一个学习器。对 threshold 选中的每个
三维 grid cell，再加入一层六连通邻居：

```text
(x-1, y,   z)   (x+1, y,   z)
(x,   y-1, z)   (x,   y+1, z)
(x,   y,   z-1) (x,   y,   z+1)
```

即沿 `±x、±y、±z` 各扩张一步；存在于 cell schema 中的邻居均提升为 E3。其作用是用受控
冗余保护预测边界，减少 500 ms 预测误差造成的漏传。它将 recall 提高到 85.61%，代价是
precision 与 F1 下降。

## 7. 最终评测结果

### 7.1 Cell visibility 与最终 policy

最终 `threshold=0.20 + guard6` 在 1,339,994 个 cell-frame 样本上的结果：

| Metric | LR-V1 |
|---|---:|
| MSE against fraction target | 0.03955 |
| Accuracy | 0.71570 |
| Precision | 0.19394 |
| Recall | **0.85615** |
| F1 | 0.31624 |
| True positive | 88,095 |
| False positive | 366,154 |
| False negative | 14,802 |
| True negative | 870,943 |

LR-V1 有意通过低阈值和 guard6 提高覆盖率；较低的 precision/F1 反映 safety margin 带来的
冗余 cell。MSE 衡量连续 LR score，分类指标衡量最终 guard6 policy。

### 7.2 Gaussian point count

| Representation | Total points | Mean points/frame |
|---|---:|---:|
| LR-V1 | 48,201,783 | 14,172.83 |
| Full E3 | 63,526,360 | 18,678.73 |
| DanceNet3D GT | 212,679,616 | 62,534.44 |

- Gaussian-count reduction vs Full E3：**24.12%**。
- Gaussian-count reduction vs DanceNet3D GT：**77.34%**。

### 7.3 Bandwidth

- Equivalent policy rate：989.83 Mbps。
- Byte reduction vs Full E3：**25.49%**。
- Byte reduction vs DanceNet3D GT PLY：**72.06%**。

Mbps 是未压缩 PLY record 在 30 FPS 下的等效数据率，不代表最终网络 codec bitrate。点数
减少与字节减少必须分别报告，因为不同 representation 的每点属性和存储开销不同。

### 7.4 Full-frame QoE

| Metric | LR-V1 |
|---|---:|
| PSNR vs GT | **28.768 dB** |
| Full E3 PSNR vs GT | 29.265 dB |
| PSNR delta vs Full E3 | **-0.498 dB** |
| SSIM vs GT | **0.97763** |
| LPIPS Alex vs GT (lower is better) | **0.03756** |

### 7.5 Foreground QoE

| Metric | LR-V1 |
|---|---:|
| Foreground PSNR | 18.869 dB |
| Foreground SSIM | 0.90936 |
| Foreground LPIPS Alex (lower is better) | 0.14589 |

## 8. 推荐汇报表述

> 为减少 500 ms 预测下未来可见区域的漏传，并避免画面空洞和显著视觉伪影，LR-V1 有意
> 采用较低的决策阈值和六连通空间 guard band，以一定冗余传输换取更高覆盖率。最终可见
> cell recall 达到 85.61%；尽管 precision 和 F1 分别下降至 19.39% 和 31.62%，该策略仍比
> Full E3 减少 24.12% 的 Gaussian points 和 25.49% 的传输字节，同时将全景 PSNR 损失
> 控制在约 0.50 dB。

## 9. 当前适用范围

- 数据只有 10 条 trace，且全部为 CircleTurns。
- 当前结论尚不能证明跨用户或跨运动模式的泛化能力。
- Gaze 对普通 threshold policy 的收益较弱；当前主要收益出现在 guard6 高质量 operating point。
- Guard6 是固定空间安全边界，尚未根据运动速度、方向或 cell 成本自适应。
