# Linear Regression FoV Predictor: Living Design Record

> 本文档是 LR FoV prediction 的持续设计记录。任何改变输入、标签、特征、
> 训练划分、模型、阈值、policy 后处理、评测方式或 HPC 路径的提交，都必须
> 同步修改本文档的相应章节和末尾 Change log。

## 1. 一句话定义

当前已经完成正式 BNQ 的机制是 head-only；新增、待 HPC 验证的 gaze
iteration 为：

```text
过去 500 ms 的 6DoF + gaze direction
        ↓
固定的 motion-aware 特征变换
        ↓
一个多输出 Ridge Linear Regression
        ↓
500 ms 后每个 cell 的 visibility score
        ↓
threshold / 可选空间 guard band
        ↓
该 cell 发送 Base-only 或 Base + E3
```

Head-only 版本中“6DoF 直接预测 cell visibility”是正确的。Gaze 版本则是
“历史 6DoF + 历史 gaze 直接预测 cell visibility”。模型没有先预测未来
DoF 再调用几何可见性计算，而是一步映射到未来所有 cell 的 score。训练
visibility 是监督标签；正式 LR 推理不读取当前或未来 visibility，gaze 也
只来自当前时刻及之前。

## 2. 数据契约

### 输入 trace

- 数据：10 条 `BiancaGolden_CircleTurns` trace。
- 字段：位置 `LocationX/Y/Z` 和旋转 `RotationRoll/Pitch/Yaw`。
- 重采样：30 FPS。
- Euler 角在插值前使用 `unwrap` 消除跨越 ±180 度造成的跳变。
- History：500 ms；代码使用 15 个历史间隔并包含当前采样，因此共 16 个
  6DoF pose。

### 监督标签

- 每个 trace 对应一份逐帧、逐 cell 的 GT visibility CSV。
- 原始标签是 `contributing_gaussian_fraction ∈ [0,1]`。
- 当前正式 500 ms 模型使用 binary target：

```text
fraction >= 0.5  -> 1 (future-visible / should receive E3)
fraction <  0.5  -> 0
```

- CSV 中缺失的 cell 在该帧记为 0。
- 输出 cell schema 是 10 份 visibility CSV 的 cell ID 并集。
- 时间对齐按 timestamp 寻找最接近 `current + 500 ms` 的有效 visibility
  frame；允许的最大误差为约 1.1 个采样间隔。

### Gaze 输入

- Trace 已包含 `GazeHitX/Y/Z` 和 `GazeConfidence`。
- `GazeHitXYZ` 实测模长约为 1，因此按 world-space unit direction 处理，而
  不是三维 hit position。
- 正式 10 条 trace 中 `confidence >= 0.5` 且 gaze 非零的样本占 98.49%；
  有效 confidence 约为 0.998。
- 起始约 1.14 秒 gaze 尚未初始化。Gaze LR 和其所有公平对照都要求完整
  500 ms gaze history 有效；无效窗口直接跳过，不把零向量当正常 gaze。
- 有效 gaze 在插值后重新单位化。

### 严禁的数据泄漏

Head-only LR inference 只能来自历史 6DoF；gaze iteration 只能额外读取历史
gaze direction。以下数据不得成为 LR feature：

- 当前 cell visibility；
- 未来 cell visibility；
- 测试 trace 的标签统计；
- 测试集上选择出的 threshold。

`persistence` 会读取当前 visibility，但它只是独立的 causal baseline，不是
LR 的输入。单元测试会修改 future visibility label 并验证 linear policy
decision 不变。

## 3. Trace-level 8:2 划分

使用固定 seed `20260731`，按完整 trace 文件划分，避免相邻帧泄漏。

训练 trace（8）：

```text
26_7_29_12_33_39
26_7_29_12_35_7
26_7_29_12_40_25
26_7_31_14_59_37
26_7_31_15_3_19
26_7_31_15_5_13
26_7_31_15_6_30
26_7_31_15_7_7
```

测试 trace（2）：

```text
26_7_29_12_37_21
26_7_31_15_1_21
```

在 8 条训练 trace 内部再固定划分为 6 条 threshold-fit 和 2 条 calibration
trace：临时模型只用于选择 decision threshold；最终模型在全部 8 条训练
trace 上重新拟合。两条正式测试 trace 不参与 threshold calibration。

## 4. 两代 LR feature

### V1: `raw_history`

把 16 个 pose 直接展开：

```text
16 poses × 6 DoF = 96 dimensions
```

这是得到当前已完成 BNQ 结果的 baseline 模型。

### V2: `motion_quadratic`

V2 不改变学习器，仍然只有一个 Ridge LR；它只加入固定、确定性的 feature
engineering，以便显式表达 500 ms 运动趋势。

| Feature group | 维度 |
|---|---:|
| 当前 pose | 6 |
| 前 15 个 pose 相对当前 pose 的差 | 90 |
| 500 ms 长期速度 | 6 |
| 最近 100 ms 短期速度 | 6 |
| 由相邻短期速度得到的加速度 | 6 |
| 常加速度外推的 500 ms future pose | 6 |
| future pose 的平方和两两交互项 | 21 |
| 当前 pose × 短期速度 | 6 |
| 总计 | 147 |

外推只用于构造 LR feature：系统没有把外推 pose 送入 renderer，也没有通过
几何算法从 pose 计算 cell；最终仍然由 LR 一步输出 cell scores。

二阶项使它也可以称为 polynomial Ridge regression，但所有基函数都是固定
的，唯一学习的部分仍是线性系数，因此没有跳出 LR 范畴。

### V3: `raw_gaze`（待 HPC 评测）

为了单独测量 gaze 的增益，V3 不叠加表现不稳定的 V2 head-motion feature。
它保留 V1 的 96 维 raw head history，并加入 108 维 gaze feature：

| Feature group | 维度 |
|---|---:|
| V1 raw head history | 96 |
| 当前 world-space gaze | 3 |
| 前 15 个 gaze 相对当前 gaze 的差 | 45 |
| gaze 长期/短期速度 | 6 |
| gaze 加速度 | 3 |
| 单位化的 500 ms gaze 外推方向 | 3 |
| 16 帧 gaze 在 head forward/right/up basis 中的坐标 | 48 |
| 总计 | 204 |

Head basis 使用 trace 的 pitch/yaw 构造；样本检查表明其 forward 与记录的
world gaze 坐标方向一致。Batch 会在相同 gaze-valid history 窗口重新训练
V1 baseline；V1 和 V3 均使用 `alpha=1.0`、相同 binary target 和相同 trace
split，因此主要差异是 gaze feature。

## 5. 学习器

设 feature matrix 为 `X`，所有 cell target 为 `Y`：

1. 每个 X 维度用训练数据的 mean/std 标准化；近零 std 设为 1。
2. 每个输出 cell 的 Y 也分别标准化，使低频 cell 不会只因方差尺度小而被
   完全忽略。
3. 求解多输出 Ridge：

```text
W = (XᵀX + αI)⁻¹ XᵀY
```

4. 推理后恢复 Y 的尺度，并把 score clip 到 `[0,1]`。

V1 正式模型使用 `α=1.0`；V2 BNQ iteration 使用 `α=3.0`。模型保存为 NPZ，
包括 feature mean/scale、target mean/scale、系数、cell IDs、feature mode、
target mode 和校准 threshold。

V3 gaze model 使用 `α=1.0`，保存的 input contract 是
`6dof_and_gaze_history`。

代码入口：

- feature/alignment/training：`src/fovsim/prediction.py`
- decision adapter：`src/fovsim/predicted_policy.py`
- CLI：`python -m fovsim predict-linear` 和
  `python -m fovsim predict-linear-policy`

## 6. Score 到 streaming policy

默认 linear policy 对每个 cell 做：

```text
score >= decision_threshold -> target_level = 3 (Base + E3)
score <  decision_threshold -> target_level = 0 (Base only)
```

V1 通过训练集内部最大化 F2 选出的正式 threshold 是 `0.20`。F2 比 precision
更强调 recall，符合 streaming 中漏发可见 cell 通常比多发 cell 更危险的
目标。

V2 BNQ sweep 不在测试集上选单一 threshold，而是把 `0.10/0.15/0.20/0.25`
作为 rate-quality operating points 全部报告。这样可以画 rate-distortion
curve，避免先看测试结果再偷偷改变唯一 threshold。

可选 `guard_band_steps=1` 使用 cell grid 的六连通邻居（±x、±y、±z）扩张
一次。它不是另一个学习器，而是 streaming safety margin；用于测试以额外
带宽减少 cell miss 是否能换回明显 QoE。

## 7. Baselines 与 BNQ

完整 V2 batch 对每条测试 trace 评测：

```text
base_only
persistence
lr_v1_t020
lr_v2_t010
lr_v2_t015
lr_v2_t020
lr_v2_t025
lr_v2_t020_guard6
```

QoE evaluator 对每个 operating point 构造 Base + selected E3，并同时渲染
Full E3 和 DanceNet3D GT。最终汇总：

- selected cells/frame；
- raw-Ply-equivalent Mbps；
- bandwidth saving vs Full E3；
- data reduction vs DanceNet3D GT PLY；
- full-frame PSNR/SSIM/LPIPS；
- foreground PSNR/SSIM/LPIPS；
- PSNR delta vs Full E3。

绝对 Mbps 是未压缩 PLY record 在 30 FPS 下的等效数据率，不应当解释为最终
网络 codec bitrate；主要比较量是同一资产格式下的相对节省。

一键 HPC batch：

```bash
bash scripts/submit_lr500_bnq_batch.sh
```

输出根目录：`/scratch/$USER/fov_lr500_bnq_v2`；最终表格：
`bnq_summary.csv`，完整记录：`bnq_summary.json`。

## 8. 当前已知结果（V1）

Cell prediction（两条 test trace 合并）：

- threshold：0.20；
- recall：0.6042；precision：0.3757；
- F1：0.4633；F2：0.5386；
- target cells/frame：30.26；predicted cells/frame：48.66；
- missed cells/frame：11.98；extra cells/frame：30.38。

End-to-end BNQ：

- LR policy vs GT：27.37 dB PSNR、0.9752 SSIM、0.0417 LPIPS；
- 相比 Full E3：约 1.97 dB PSNR 损失；
- bandwidth saving vs Full E3：45.93%；
- data reduction vs DanceNet3D GT PLY：79.73%。

### V2 motion-aware LR 结果

V2 在训练集内部校准得到 threshold `0.18`。两条 test trace 合并的 cell
prediction 为：

- recall：0.6767；precision：0.3077；
- F1：0.4230；F2：0.5458；
- predicted cells/frame：66.55；
- missed cells/frame：9.78；extra cells/frame：46.08。

与 V1 threshold 0.20 相比，V2 recall 提高约 7.25 个百分点、每帧少漏约
2.19 个 cell，但每帧多产生约 15.70 个 extra cell；F2 只从 0.5386 增至
0.5458。因此 V2 更积极，但 cell-level rate efficiency 没有明显改善。

完整 BNQ operating points：

| Variant | Mbps | Saving vs Full E3 | PSNR vs GT | ΔPSNR vs Full E3 |
|---|---:|---:|---:|---:|
| Base-only | 269.64 | 79.66% | 25.77 dB | -3.52 dB |
| Persistence | 529.04 | 60.09% | 26.62 dB | -2.66 dB |
| V1 t=0.20 | 716.71 | 45.93% | 27.33 dB | -1.96 dB |
| V2 t=0.10 | 999.14 | 24.62% | 28.19 dB | -1.10 dB |
| V2 t=0.15 | 894.59 | 32.51% | 27.63 dB | -1.66 dB |
| V2 t=0.20 | 802.62 | 39.45% | 27.19 dB | -2.10 dB |
| V2 t=0.25 | 726.66 | 45.18% | 26.90 dB | -2.39 dB |
| V2 t=0.20 + guard6 | 1076.48 | 18.79% | 28.78 dB | -0.51 dB |

V2 t=0.20 和 t=0.25 均被 V1 t=0.20 支配：它们使用更多带宽但 PSNR 更低。
因此不能宣称 motion-aware feature 本身全面优于 V1。当前观测到的 Pareto
序列是 Base-only、Persistence、V1 t=0.20、V2 t=0.15、V2 t=0.10、V2
guard6、Full E3。Guard6 是最接近 Full E3 的 streaming 点，但其改善主要
来自空间 safety margin，不能归因于 LR feature 本身。

### Gaze matched BNQ 结果

新 batch 在完全相同的有效 gaze 时间区间比较：Base-only、Persistence、
V1 与 raw-gaze LR 的 threshold `0.10/0.15/0.20/0.25`，以及双方的
threshold 0.20 + guard6。共 12 个 variants × 2 test traces = 24 个 GPU
tasks。

```bash
bash scripts/submit_lr500_gaze_bnq_batch.sh
```

输出根目录：`/scratch/$USER/fov_lr500_gaze_bnq_v1`。

所有 variants 使用相同的 3401 帧 gaze-valid test interval。完整结果：

| Variant | Mbps | Saving vs Full E3 | PSNR vs GT | ΔPSNR vs Full E3 |
|---|---:|---:|---:|---:|
| Base-only | 270.18 | 79.66% | 25.74 dB | -3.52 dB |
| Persistence | 528.41 | 60.23% | 26.60 dB | -2.67 dB |
| V1 t=0.10 | 947.62 | 28.67% | 28.20 dB | -1.07 dB |
| V1 t=0.15 | 824.60 | 37.93% | 27.69 dB | -1.57 dB |
| V1 t=0.20 | 718.71 | 45.90% | 27.28 dB | -1.98 dB |
| V1 t=0.25 | 629.87 | 52.59% | 26.88 dB | -2.39 dB |
| Gaze t=0.10 | 917.24 | 30.96% | 28.06 dB | -1.21 dB |
| Gaze t=0.15 | 802.60 | 39.59% | 27.59 dB | -1.68 dB |
| Gaze t=0.20 | 705.73 | 46.88% | 27.17 dB | -2.10 dB |
| Gaze t=0.25 | 627.78 | 52.75% | 26.90 dB | -2.36 dB |
| V1 t=0.20 + guard6 | 1025.25 | 22.83% | 28.79 dB | -0.47 dB |
| Gaze t=0.20 + guard6 | 989.83 | 25.49% | 28.77 dB | -0.50 dB |

同 threshold 下，Gaze LR 通常减少约 2--30 Mbps，但在 t=0.10--0.20
降低约 0.10--0.14 dB PSNR；按 V1 curve 插值到相同 bitrate 后，两者差异
大致只有 0.02--0.06 dB。Gaze t=0.25 以略低带宽取得高 0.024 dB PSNR，
是一个很小的严格改进，但不足以证明普通 Gaze LR 全面优于 V1。

最有价值的是 guard6：Gaze guard 相比 V1 guard 少 35.41 Mbps（约 3.45%），
PSNR 只低 0.024 dB，同时 SSIM/LPIPS 略好。它相对 Full E3 节省 25.49%，
仅损失 0.50 dB，是当前最合理的高质量 gaze operating point。谨慎结论是：
gaze 对普通 threshold policy 的 matched-rate 收益很弱，但可帮助 guard-band
policy 用近似相同 QoE 进一步减少少量带宽。

BNQ 汇总优先使用每份已完成 QoE `summary.json` 中记录的真实 GT sequence
路径，不依赖可能被外部项目设置为父目录的通用 `GT_ROOT`。如果统计时某个
GT PLY 确实已被删除，其他 QoE/带宽字段仍会正常输出；相对 GT 的节省字段
留空，并在 `gt_size_status`/`missing_gt_asset_ids` 中明确标记，避免用不完整
分母产生错误百分比。

## 9. 已知限制

- 只有 10 条、且全部是 CircleTurns，跨用户和跨运动类型泛化尚未验证。
- Euler unwrap 解决了角度跳变，但没有使用 quaternion 表达。
- 直接多输出 LR 对复杂 cell visibility 边界的表达能力有限。
- 所有 cell 的回归损失没有直接按 Gaussian bytes 或画面贡献加权。
- Threshold calibration 优化 F2，不等同于直接优化 PSNR/bitrate。
- Guard band 是固定六邻域，没有根据速度或 cell 传输成本自适应。
- Gaze LR 的普通 threshold matched-rate 收益很弱；目前主要价值出现在
  guard6 高质量点，仍需更多非规则转头 trace 验证泛化。
- DanceNet3D GT 是参考 PLY，不是可直接比较的网络 codec。

## 10. LR 变更强制检查清单

任何 LR 相关提交必须检查并按需更新本文档：

- [ ] inference 输入是否仍然只有历史 6DoF？
- [ ] history、FPS 或 horizon 是否改变？
- [ ] feature mode/维度/公式是否改变？
- [ ] target 定义或 visibility threshold 是否改变？
- [ ] trace split、seed 或 calibration 数据是否改变？
- [ ] Ridge alpha、标准化或求解方式是否改变？
- [ ] decision threshold 或 guard policy 是否改变？
- [ ] cell-level metrics 是否更新？
- [ ] BNQ/GT 结果是否更新？
- [ ] HPC 路径、脚本或输出 schema 是否改变？
- [ ] 在下方 Change log 添加 commit/date/原因。

## 11. Change log

| Date | Commit | Change |
|---|---|---|
| 2026-07-31 | `09ad493` | 加入训练集内部 decision-threshold calibration。 |
| 2026-07-31 | `ff346b3` | 改为 F2/recall-oriented threshold，并报告 miss/extra cells。 |
| 2026-07-31 | `c4b1ec7` | 固定 500 ms binary-target Ridge iteration 与自动评估。 |
| 2026-07-31 | `fefee92` | 将 LR decisions 接入 bandwidth 与 QoE renderer。 |
| 2026-08-01 | `df9bb8c` | 加入 147 维 motion-aware quadratic LR、threshold sweep、Base/Persistence 和完整 BNQ batch。 |
| 2026-08-01 | `abceb80` | 建立本文档，作为后续 LR 变更的强制追溯记录。 |
| 2026-08-01 | `cd69e48` | 记录 V2 cell prediction、完整 BNQ curve、Pareto 与 dominated operating points。 |
| 2026-08-01 | `b8197dd` | 加入 204 维 raw-head + gaze Ridge LR、有效 gaze 窗口约束及 matched BNQ batch；结果待 HPC。 |
| 2026-08-01 | `d7a2527` | 修复 BNQ summary 继承通用 `GT_ROOT` 父目录的问题，并允许缺失 GT-size 时保留其他汇总指标。 |
| 2026-08-01 | this commit | 记录 gaze matched BNQ：普通 threshold 近似持平，gaze guard 以近似 QoE 额外减少 35.41 Mbps。 |
