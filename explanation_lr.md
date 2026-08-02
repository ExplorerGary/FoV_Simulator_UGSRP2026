# Linear Regression FoV Predictor: Living Design Record

> 本文档是 LR FoV prediction 的持续设计记录。任何改变输入、标签、特征、
> 训练划分、模型、阈值、policy 后处理、评测方式或 HPC 路径的提交，都必须
> 同步修改本文档的相应章节和末尾 Change log。

## 1. 一句话定义

当前归档的 LR-V1 是已经完成正式 BNQ 的 raw-head + gaze + guard6 机制：

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
- `cellsight-cp` 当前实现按 CellSight 的圆周表示处理 Euler 角：Roll、Pitch、Yaw
  分别转成 `sin/cos`，在单位圆上插值、重新归一化，再用 `atan2` 恢复角度。
  进入 LR 时不使用恢复后的裸角度，而使用每个角的 `sin/cos` 对。
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

### V3: `raw_gaze`

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

### CellSight rotation preprocessing（`cellsight-cp`，待重新训练）

该分支保持 6DoF 的语义不变，但改变旋转在重采样和 LR feature 中的数值表示：

```text
每个源 Euler 角 θ（degrees）
  -> (sin θ, cos θ)
  -> 分别沿时间插值
  -> 将插值后的二元组重新归一化到单位圆
  -> atan2(sin θ, cos θ) 仅供对齐、渲染和日志使用

LR pose feature
  = [x, y, z,
     sin roll, cos roll,
     sin pitch, cos pitch,
     sin yaw, cos yaw]
```

因此每个 pose 的 LR 数值维度由 6 变为 9。在 30 FPS、500 ms history（包含
当前帧，共 16 个 pose）下，`raw_history` 从 96 维变为 144 维，`raw_gaze`
从 204 维变为 252 维，`motion_quadratic` 从 147 维变为 234 维。学习器、
target、trace split、threshold 和 policy 均未在本次改动中改变。

这一处理避免 `179° -> -179°` 被误认为约 358° 的大幅运动。模型 NPZ 新增
`rotation_encoding=per_angle_sin_cos`；旧模型必须重新训练，policy loader 会
明确拒绝缺少该契约的旧模型，避免把旧权重误用于新 feature schema。以上仅是
已实现的数据处理变更，尚无新的 HPC accuracy 或 BNQ 结果。

### CellSight-style DoF-to-DoF LR（2026-08-02，本地已运行）

该步进实验把 target 从 cell visibility 临时切换为未来 6DoF，用来单独验证
trajectory prediction。它不替代既有 DoF-to-cell streamer，也尚未接入 BNQ。

固定实验契约：

```text
timeline       = 30 FPS
history        = 500 ms = 15 intervals = 16 samples including current
horizon        = 100 ms = 3 frames
split          = the existing trace-level 8:2 split, seed 20260731
training       = 8 complete CircleTurns traces, 11556 windows
testing        = 2 complete CircleTurns traces, 3607 windows
environment    = local conda gs_train (Python 3.11.15, sklearn 1.5.2)
```

严格采用论文中 LR baseline 的独立坐标设计，不加入 gaze、速度、加速度、二次项
或 guard：`x/y/z` 各自以 16 个同坐标历史值拟合一个普通
`LinearRegression(fit_intercept=True)`；每个 Euler 角拆为 sin/cos，六个圆周坐标
同样各自独立拟合，最后单位化并用 `atan2` 恢复角度。因此一共训练 9 个相互独立
的 OLS，模型不共享不同 DoF 的信息。Step 0 是未来 pose 等于当前 pose 的
Persistence；Step 1 是该 CellSight-style LR。

两条测试 trace 合并结果：

| Metric | Step 0 Persistence | Step 1 DoF LR | LR change |
|---|---:|---:|---:|
| Position MSE (cm²) | 22.015842 | 18.232571 | -17.18% |
| Position RMSE (cm) | 4.692104 | 4.269961 | -9.00% |
| Position MAE (cm) | 1.521023 | 1.273248 | -16.29% |
| Position R² | 0.998783 | 0.998992 | +0.000209 |
| Orientation MSE (degree²) | 6.558731 | 2.798082 | -57.34% |
| Orientation RMSE (degree) | 2.561002 | 1.672747 | -34.68% |
| Orientation MAE (degree) | 1.470996 | 1.064244 | -27.65% |
| Orientation circular R² | 0.997521 | 0.998939 | +0.001418 |

R² 的主旋转指标在 sin/cos 圆周域计算，避免 ±180° 边界虚增方差；位置使用普通
coefficient of determination。逐 DoF MSE/R²、逐 trace 指标、模型系数分别保存为：

```text
outputs/dof_lr_500ms_to_100ms/dof_lr_evaluation.json
outputs/dof_lr_500ms_to_100ms/step_metrics.csv
outputs/dof_lr_500ms_to_100ms/per_trace_metrics.csv
outputs/dof_lr_500ms_to_100ms/dof_lr_model.npz
```

LR 在两条测试 trace 上都优于 Persistence；因此本轮没有启动“失败后”的超参数
复制。论文的 LR 超参数只有 LR30/LR90 history window，而用户要求保持 500 ms
输入；GraphGRU 的 learning rate、hidden dimension、epoch 等不适用于普通 OLS，
也没有混入本实验。

### CellSight LR30/LR90 source-compatible sweep（2026-08-02）

进一步核对仓库后，CellSight 的 LR30/LR90 是每个预测窗口内部以时间索引拟合的
局部 OLS，不是跨训练 trace 拟合并保存的模型。`LR30`/`LR90` 分别表示严格的
30/90 个历史样本；每个位置与每个角度的 sin/cos 坐标独立拟合。为公平比较，
本地 sweep 只评测两种窗口都已完整建立的共同 target frames，所以同一 horizon
下两者样本数及 Persistence 完全一致。

| Horizon | Model | Position MSE (cm²) | Position R² | Rotation MSE (degree²) | Circular R² |
|---:|---|---:|---:|---:|---:|
| 33 ms | LR30 | 12.082236 | 0.999301 | 20.069694 | 0.992583 |
| 33 ms | LR90 | 72.026273 | 0.995832 | 89.722675 | 0.969255 |
| 100 ms | Persistence | 3.968973 | 0.999770 | 6.700300 | 0.997495 |
| 100 ms | LR30 | 20.999782 | 0.998783 | 33.640951 | 0.987711 |
| 100 ms | LR90 | 86.761286 | 0.994972 | 106.652599 | 0.963921 |
| 333 ms | LR30 | 70.829064 | 0.995875 | 102.960223 | 0.964655 |
| 333 ms | LR90 | 148.620250 | 0.991345 | 169.541145 | 0.944878 |

CircleTurns 上 LR30 在所有三个 horizon 都优于 LR90，与论文 Tables 2/3/5--10
的短期排序一致。论文的数值是预测 DoF 经过相机/scene geometry 后得到的 per-cell
viewport overlap、angular span 或 occlusion visibility R²；本表是上游 DoF R²，
数据集也不同，因此数值不应相等。论文 8i viewport-overlap R² 在 33/333 ms
分别为 LR30 `0.941/0.809`、LR90 `0.824/0.718`；本地 DoF circular R²
为 `0.992583/0.964655` 与 `0.969255/0.944878`，确认了 LR30 > LR90 及 horizon
变长时 R² 下降的共同趋势，但不能宣称完成 cell-level 数值复现。

源码还显示，8i trajectory generator 对位置使用真实 horizon，却把 orientation
调用硬编码为 `future_steps=1`；FSVVD 路径则对全部九个编码坐标使用真实 horizon。
本地 sweep 采用论文方法定义及 FSVVD 的一致行为，对位置和旋转都使用真实 horizon，
没有复制 8i 脚本中的不一致。

### DoF-to-cell geometry integration study

CellSight trajectory baseline 的 DoF-to-cell 不是第二个学习器，而是确定性几何：
预测 pose 生成 camera matrix；固定场景内均匀采样 10,000 点，按 cell 统计投影进
viewport 的比例；occlusion-aware 版本还对未来 point cloud 进行 FoV crop 和
Open3D HPR，再按 cell 统计可见点比例。完整接入分析记录在
`docs/cellsight_dof_visibility_integration.md`。

该机制可以接入现有 pipeline：预测 DoF 后生成 per-cell score，再写入
`cell_decisions.csv`，现有 bandwidth/QoE evaluator 无需改变。但 CellSight 的
viewport-overlap/HPR fraction 与当前 renderer-based
`contributing_gaussian_fraction` 定义不同，必须分字段、重新校准 threshold，不能
直接冒充现有 visibility。建议先实现 CellSight overlap adapter 做论文同口径 R²，
再让现有 Gaussian visibility generator 接受 predicted-pose override，得到可直接
用于正式 BNQ 的 renderer-consistent score。

### Visibility score-definition oracle BNQ sweep（待 HPC 结果）

为回答“发送 E3 的 fraction 是否等同于 CellSight visibility”，新增一个严格标记
为 oracle 的 score-definition sweep。它先使用真实 pose 产生的已有 visibility CSV，
只比较 score 定义及 selection rule，不声称是 100 ms prediction：

| Family | Score | Operating points |
|---|---|---|
| Gaussian occlusion/compositing-aware | `contributing_gaussian_fraction` | absolute threshold 0.10/0.20/0.30/0.50 |
| Gaussian viewport/no-occlusion proxy | `rasterized_gaussian_count / active_gaussian_count` | absolute threshold 0.10/0.20/0.30/0.50 |
| Screen contribution | `image_share` | descending cells until cumulative 0.90/0.95/0.99 |

第二项只是 CellSight viewport-overlap `F` 的 Gaussian analogue，不是论文用均匀空间
采样得到的严格 volume fraction；第一项也不是 HPR point fraction，而是更符合当前
3DGS renderer 的 `T*alpha` contribution fraction。第三项跨 cell 总和约为 1，更适合
累计覆盖规则，不适合与前两项共用绝对 threshold。

正式 batch 共 11 variants × 2 test traces = 22 个完整 BNQ jobs，输出相同的 cell
classification、selected cells、Gaussian point counts、bytes/Mbps、PSNR、SSIM、
LPIPS 和 foreground metrics。由于同数值 threshold 不代表同 rate，结果分析必须同时
绘制 raw threshold curve，并在 matched mean-selected-cells / matched Mbps 下比较。

本地 30 帧 sanity check 只验证计算和 schema，不能作为正式结论：

```text
contributing threshold 0.20 -> 10.60 selected cells/frame
rasterized threshold 0.20   -> 14.13 selected cells/frame
image-share coverage 0.95   ->  7.30 selected cells/frame
```

提交入口为 `scripts/submit_visibility_score_bnq_batch.sh`，默认独立输出目录
`/scratch/$USER/fov_visibility_score_bnq_v1`。完成该 oracle study 后，再把相同 policy
generator 的输入替换成 predicted-pose visibility，以隔离 score-definition 与 DoF
prediction 两类误差。

首次 HPC 提交的 22 个 policy CSV 均成功生成，但 QoE evaluator 因 wrapper 未把
`$REPO/src` 加入容器内 `PYTHONPATH`，统一报错 `ModuleNotFoundError: fovsim`。后续
提交已在调用 evaluator 前显式 export repo source path；该失败没有产生 QoE 结果，
不属于实验数值。

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

### LR-V2 100 ms iteration（已完成）

本轮只把 prediction horizon 从 500 ms 缩短到 100 ms；history 仍为 500 ms（16 帧），以隔离
预测距离变化的影响。`raw_gaze` 中的 gaze extrapolation 使用实际 `horizon_s=0.1`，因此不会
错误地用 500 ms gaze 外推预测 100 ms target。模型仍是 `alpha=1.0` 的多输出 Ridge LR，
binary target、8:2 trace split 与 gaze-valid matched window 均保持不变。

训练集内部同时产生两个 threshold：

- `f2`：在 calibration traces 上最大化 F2，并保存进 NPZ；
- `safe`：在 calibration recall 至少为 0.85 的候选中选择最高 threshold，从而在满足覆盖率
  约束时尽量少传 cell。若搜索范围内无法达到 0.85，则明确标记 `constraint_met=false`，使用
  calibration recall 最高的候选继续完成评测，避免丢失整组 BNQ 结果。

完整 batch 评测 Base-only、100 ms Persistence、Head/Gaze F2、Head/Gaze safe，以及 Head/Gaze
F2 + guard6，共 8 variants × 2 test traces。最终汇总自动报告 cell MSE/precision/recall/F1/F2、
Gaussian-count reduction、byte reduction、full-frame 和 foreground QoE。

结果显示，100 ms Persistence 的 MSE/F1 为 0.02284/0.7575，并以 593.86 Mbps 达到
27.76 dB；它在带宽和 PSNR 上都严格支配普通 Head/Gaze F2 LR。两种 safe calibration 均未在
搜索范围内达到 0.85 recall，回退至 threshold 0.10。最高画质点是 Gaze F2 + guard6：recall
0.8874、1045.22 Mbps、28.84 dB，相比 Full E3 节省 21.18% bytes、20.14% Gaussian points，
PSNR 损失 0.40 dB。它比 500 ms LR-V1 多约 55.4 Mbps，只提高约 0.075 dB，因此没有取代
LR-V1 成为主要 rate-quality operating point。

```bash
bash scripts/submit_lr100_bnq_batch.sh
```

默认输出目录：`/scratch/$USER/fov_lr100_bnq_v1`。

### 100 ms current-visibility LR iteration（待 HPC 结果）

为利用 100 ms 下最强的短期相关性，新增 feature mode
`raw_gaze_current_visibility`：在原 204 维 raw-head + gaze feature 后追加预测时刻 `t` 的完整
cell visibility fraction 向量。输出仍是 `t+100 ms` 的所有 cell visibility，唯一学习器仍是
标准化多输出 Ridge LR（`alpha=1.0`）。该机制明确允许读取当前 visibility，但禁止读取未来
visibility；因此它应与 Persistence 比较，并与不读取当前 visibility 的 LR-V1 分开描述输入条件。

新 batch 在完全相同 gaze-valid window 比较 Base-only、Persistence、matched Gaze LR、
current-visibility LR 的 F2/safe thresholds，以及双方 F2 + guard6。Threshold 搜索扩展到
0.01--0.50，以免人为阻止 recall-constrained operating point。完整 cell、point-count、bandwidth
与 QoE 结果写入独立目录，不覆盖上一轮：

```bash
bash scripts/submit_lr100_currentvis_bnq_batch.sh
```

默认输出目录：`/scratch/$USER/fov_lr100_currentvis_bnq_v1`。该实验尚未运行，不能预先声明
current-visibility LR 超过 Persistence。

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
| 2026-08-01 | pending | 将 `raw_gaze`、threshold 0.20、guard6 operating point 归档为 `LR-V1`；记录最终 guard-policy 分类指标和 Gaussian-count reduction。 |
| 2026-08-01 | pending | 加入 LR-V2 100 ms candidate：保留 500 ms history，新增 recall-constrained threshold，并提交 Head/Gaze、Persistence、guard6 的完整 BNQ dependency chain；结果待 HPC。 |
| 2026-08-01 | pending | 记录 100 ms BNQ 结果；新增 causal current-visibility + raw-gaze Ridge feature mode、泄漏测试与完整 matched BNQ batch；新结果待 HPC。 |
| 2026-08-01 | pending | `cellsight-cp` 将 Euler 重采样和 LR 旋转输入改为 CellSight 式单位圆 `sin/cos` 表示；新增 wrap-around 测试和模型 rotation schema，尚未重新训练。 |
| 2026-08-02 | pending | 新增并本地运行 30 FPS、500 ms history、100 ms horizon 的独立坐标 DoF-to-DoF OLS；报告 Persistence→LR 的逐维 MSE/RMSE/MAE/R² 与圆周 R²。 |
| 2026-08-02 | pending | 按源码语义加入 LR30/LR90 窗口内 OLS sweep，统一共同评测帧并运行 33/100/333 ms；记录论文表格口径差异与 DoF-to-cell geometry 接入设计。 |
| 2026-08-02 | pending | 新增真实-pose visibility score-definition oracle sweep：contribution/rasterized absolute thresholds 与 image-share cumulative coverage，提交 22 个完整 BNQ operating points；正式 HPC 结果待运行。 |
| 2026-08-02 | pending | 修复 visibility-score BNQ 容器未继承 repo `src` 的问题，显式设置 `PYTHONPATH`；首次 22 个 tasks 在 policy 后、QoE 前失败，无实验结果。 |
