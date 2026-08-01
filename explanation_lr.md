# Linear Regression FoV Predictor: Living Design Record

> 本文档是 LR FoV prediction 的持续设计记录。任何改变输入、标签、特征、
> 训练划分、模型、阈值、policy 后处理、评测方式或 HPC 路径的提交，都必须
> 同步修改本文档的相应章节和末尾 Change log。

## 1. 一句话定义

当前正式机制是：

```text
过去 500 ms 的 6DoF
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

所以“6DoF 直接预测 cell visibility”是正确的。更精确地说，模型没有先预测
未来 DoF 再调用几何可见性计算，而是从 6DoF history 一步映射到未来所有
cell 的 score。训练 visibility 是监督标签；正式 LR 推理不读取当前或未来
visibility。

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

### 严禁的数据泄漏

正式 LR inference 的 feature 只能来自历史 6DoF。以下数据不得成为 LR
feature：

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

V2 的结果尚未产生，必须由完整 BNQ batch 得到；不能只凭 cell F2 宣称 V2
改善了最终 QoE。

## 9. 已知限制

- 只有 10 条、且全部是 CircleTurns，跨用户和跨运动类型泛化尚未验证。
- Euler unwrap 解决了角度跳变，但没有使用 quaternion 表达。
- 直接多输出 LR 对复杂 cell visibility 边界的表达能力有限。
- 所有 cell 的回归损失没有直接按 Gaussian bytes 或画面贡献加权。
- Threshold calibration 优化 F2，不等同于直接优化 PSNR/bitrate。
- Guard band 是固定六邻域，没有根据速度或 cell 传输成本自适应。
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
| 2026-08-01 | this commit | 建立本文档，作为后续 LR 变更的强制追溯记录。 |
