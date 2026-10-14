# V4 消融修复与贡献分离

本次只生成/修改代码，没有运行算法、数值等价性测试或数据集实验。数值等价性必须以你运行后生成的检查结果为准。旧 `ablation_study_v4.py` 和旧结果均保留；后续使用 **`run_ablation_v4_fixed.py`**。

## 1. 文件与计算来源

- `run_ablation_v4_fixed.py`：新运行入口、逐序列等价性门禁、评测、配对统计和旧结果比较。
- `v4_ablation_core.py`：从原始 r11 提取两个估计函数，仅加入显式消融开关；底层归一化、卷积、峰值搜索等函数和所有常数继续导入原始 `fusion_v4.py`。
- `v4_ablation_variants.py`：实验配置与预先规定的比较关系。
- `changes_from_original_core.diff`：新核心与提取原函数的文本差异，便于审核。

原始算法文件：上一级 `fusion_v4.py`；未修改。

固定来源 SHA256：`90ef2e2a840ea2a8e49a86655fb34e93fb1614c33c1f5b5376e2b9d675883b4c`。如果原版发生变化，入口会停止，不能直接跳过检查。

这里采用“原函数少量开关 + 共享底层函数 + 独立等价性门禁”，并非声称两个高层函数是同一个函数对象。后续修改原版时需要同步审阅差异，不能只更新哈希。

## 2. 修复了什么

旧 Full V4 分支直接调用原版，其余分支调用另一套实现。因此旧 Full 与主实验一致，并不能验证消融实现正确。新 Full 与所有消融配置都通过新核心运行；另调用原版作为独立参照。

保留原始数值规则：

1. `minmax` 的有效区间、区间外数值和梯度计算顺序保持原版；不提前把候选区间外清零。
2. 峰值筛选保持原版，不额外删除候选区间外的峰。
3. 候选边界、覆盖度计算、参照宽度和峰距统一保持半径 13。单偏移 d=5 只改变残差取样偏移，不改变这些条件；这不是完整的“单尺度算法”。
4. 移除参照时，参照候选不再参与两个池的打分及候选相互支持；零权重回退只在启用项内归一化，不能把被删除项重新激活。
5. 局部 odd、局部 valley、中尺度 reversal、中尺度 valley 各有独立开关。删除某项后，其相关可靠度因子也按文档定义调整；因此属于端到端模块消融。

每条序列运行时检查：

- 原版与新核心全开的全部曲线列、原版全部 metadata 字段（包括权重、亚采样焦点）一致，`atol=rtol=1e-12`。
- 所有配置的候选掩膜相同。
- 仅删除/等权组合尺度时，L/M/G 曲线、内部池权重及可靠度特征保持相同。
- 仅删除局部项时，M/G 保持相同；仅删除中尺度项时，L/G 保持相同；其他预期不变分支也逐一检查。
- 因子实验 `refs=both, scales=LMG` 的最终曲线必须与 Full 一致。

检查失败立即停止并保存失败信息，不会跳过序列后继续报告完整实验。输出目录必须是本目录下尚不存在的子目录。

## 3. 实验组

| suite | 配置数（包括 Full） | 用途 |
|---|---:|---|
| check | 4 | 全开等价性、去 M、去 G、最终尺度等权的快速验证 |
| legacy | 12 | 用修复实现重算旧 12 组，保留旧名称映射 |
| contributions | 25 | 4×4 因子组 + 纯局部/参照基线 + Full |
| components | 10 | 分开局部/中尺度物理项，以及方差和参照控制 |
| all | 45 | 一次运行上述完整集合，Full 只保留一份 |

若仍发现完整 184 条序列，`all` 应产生 8,280 条结果。脚本按实际发现的序列数核对行数，不硬编码 184；检查 manifest 中实际数量。

### 原有 12 组的解释

| 旧名称 | 新名称/准确解释 |
|---|---|
| Full V4 | Full V4：通过同一消融核心，不走原版捷径 |
| w/o Odd Parity | Joint no odd and reversal：同时删除局部 odd 和中尺度 reversal，是联合消融 |
| w/o Even Parity | No local even：删除局部 even 及相关可辨识度因子 |
| w/o Valley | Joint no local and meso valley：联合删除两处谷形项 |
| w/o Poisson Normalization | Legacy robust variance and dispersion one：残差 MAD 常方差 + dispersion 特征置 1，仍标注为联合控制 |
| w/o Information Support | No local information：删除局部信息乘子及其可辨识度作用；不是删除所有分支的活动支持 |
| Single Scale d=5 | Single offset d5 fixed radius：固定半径 13 的单偏移 |
| w/o Mesoscopic | No mesoscopic conditional：冻结 L/M/G 及原始权重，删 M 后条件归一化 |
| w/o Global CP | No global CP conditional：冻结 L/M/G 及原始权重，删 G 后条件归一化 |
| Equal Scale Fusion | Equal final scale fusion：只在最后尺度融合层等权，池内部仍自适应 |
| w/o Shared Global Reference | No shared candidate references：从两个池删除对称/事件率参照候选，保留 G 和采集长度特征 |
| Local Only | Local pooled only with references：只输出 L 池，但该池仍有共享参照，不是纯局部算法 |

旧表只用于追踪修复导致的变化，不应与新表拼接作为一套消融。旧结果对比会核对序列键、事件数量、分箱数、ROI 与 GT；这些检查仍不等同于证明原始事件字节完全相同。新运行另外保存输入计数/时长/时间戳的 SHA256。

### 分离局部、共享参照和尺度融合

16 个因子配置交叉组合：

- 参照：`none` / `sym` / `rate` / `both`。
- 尺度：`L` / `LM` / `LG` / `LMG`。

L 指局部证据池（局部 parity + trailing edge，再按开关加入参照）；M 指中尺度池；G 指全局极性变化点。

另有 8 个基线：纯信息量、纯 even、纯 odd、纯 even+odd、纯完整局部 parity，以及仅 symmetry、仅 rate、仅两个参照。纯局部系列均不包含边缘候选、参照候选、M 或 G。`Pure local parity` 仍保留信息量抑制和局部 valley；可用 `Pure local parity without valley` 单独观察 even+odd。

关键比较已写入 `paired_summary.csv`：

1. 信息量 → even/odd → even+odd → 加 valley：分离局部结构贡献。
2. 纯局部 parity → `refs=none, scales=L`：分离局部 trailing-edge 贡献。
3. 固定 scales，比较 none/sym/rate/both：分析共享参照。
4. 仅参照 both → `refs=both, scales=L`：判断局部证据在已有参照上是否仍有增益。
5. 固定 refs，比较 L/LM/LG/LMG：分析 M、G 的条件贡献。
6. Full → 最终尺度等权：分析最终自适应权重。

`factorial_interactions.csv` 给出误差的差分之差，例如
`E(LMG)-E(LM)-E(LG)+E(L)`。它反映在当前数据与固定配置下的交互，不应解释成可加的“贡献百分比”。

### 固定删除与重新加权的区别

默认尺度删除使用 **conditional**：先算原始权重 q，再把被删除尺度权重置零，归一化剩余权重；L/M/G 曲线与内部权重保持不变。因子实验在每个 refs 条件内也采用此规则。

另有 `No mesoscopic hierarchical`：把 beta 设为 0，保留 gamma，即让 L 接管原先 L/M 的总质量。这与直接条件删除不同，单列展示，不能把两者当成同一个实验。

删除内部物理项或参照时，受影响池及后续可靠度权重会重新计算，但不重新拟合参数。这类结果属于**冻结参数的端到端消融**，不是“冻结所有内部权重，仅把某项分数置零”的直接效应。

## 4. 运行指令（PowerShell）

以下指令明确使用项目已有的 `C:\Users\24686\Desktop\event focus\.venv\Scripts\python.exe`。已从该环境的安装记录确认 numpy、pandas、scipy、PyWavelets、matplotlib 均有安装记录；本次没有执行导入验证或实验。默认 `python` 可能指向缺少 pandas 的系统环境，因此不要省略解释器路径。这里不自动安装或更改环境。

先进入目录：

```powershell
Set-Location 'C:\Users\24686\Desktop\event focus\fusion_v4\Ablation study V4'
```

推荐先取每个数据集 1 条验证四个关键配置：

```powershell
& '..\..\.venv\Scripts\python.exe' .\run_ablation_v4_fixed.py --suite check --datasets SYN DAVIS EVK4 EvHF EAD --max-sequences 1 --save-curves
```

检查通过后一次完成两项优先级：

```powershell
& '..\..\.venv\Scripts\python.exe' .\run_ablation_v4_fixed.py --suite all --datasets SYN DAVIS EVK4 EvHF EAD --save-curves
```

默认数据根目录已经是 `C:\Users\24686\Desktop\event focus\_datasets`。默认输出会自动创建本目录下 `results_fixed_日期_时间_微秒`，重复运行不会覆盖旧结果。控制台最后会打印完整输出路径。

也可指定新输出目录（若已存在会拒绝运行）：

```powershell
& '..\..\.venv\Scripts\python.exe' .\run_ablation_v4_fixed.py --suite all --save-curves --output '.\results_fixed_full_01'
```

如果只想先重算旧 12 组：

```powershell
& '..\..\.venv\Scripts\python.exe' .\run_ablation_v4_fixed.py --suite legacy --save-curves
```

如果只想跑贡献分离矩阵：

```powershell
& '..\..\.venv\Scripts\python.exe' .\run_ablation_v4_fixed.py --suite contributions --save-curves
```

不要把快速检查产生的均值写进论文；正式结论使用完整数据集运行。不要把 different suite 的重复 Full 再次合并计数。

## 5. 输出文件与判读

首先检查 `run_manifest.json`：`status` 必须为 `complete`，`all_checks_passed` 必须为 true；同时核对数据集、序列总数与配置数。

- `equivalence_checks.csv`：全开、未受影响分支和因子全开锚点的逐项检查。
- `variant_definitions.csv`：所有开关、偏移和实验含义。
- `all_ablation_results.csv`：逐序列预测与误差、实际尺度权重。
- `summary_by_dataset.csv` / `summary_by_condition.csv`：数量、有效输出率、MAE、中位数、P90、最大误差。
- `paired_summary.csv`：A/B 公共有效序列上的均值差、胜/平/负、缺失数量；`B-A < 0` 表示 B 更好。
- `paired_differences.csv`：逐序列的配对差值。
- `factorial_interactions.csv`：共享参照与尺度分支的交互。
- `historical_result_comparison.csv` / `historical_summary_comparison.csv`：旧 12 组结果的配对重算比较。
- 每条序列的 `ablation_audit.json`、`all_on_equivalence.json`、`ablation_results.csv`，以及可选 `curves/*.csv`。
- `source_snapshot/` 与 manifest：本次运行代码副本、哈希、依赖版本、固定系数信息。

如失败：读取 `failure.json`、`equivalence_checks.csv`；已完成的序列会保留。失败目录中的汇总只是中间检查点，不是完整论文结果。修复后选择新输出目录重跑。

## 6. 论文中仍需保留的边界

- `valid_output_rate` 只是产生有效估计的比例，不能写成“对焦成功率”。MAE 排除缺失值，须同时看有效数量。
- SYN/DAVIS/EVK4 以时间误差 ms 为主；EvHF/EAD 以位置误差为主，严格保留协议输出的位置单位。不能把 EAD 的焦点位置单位写成微米。
- 无共享参照表示移除**候选参照证据**。所有配置仍共用全扫描增益、归一化、可观测性与长度信息，不应称为完全局部或在线算法。
- 原始 V4 系数曾在这些公开数据上联合标定。本次不重拟合系数，也不把这套消融包装成独立泛化验证。
- 为审计等价性，即使某分支被删也仍计算其上游曲线。因此本次 runtime 不适合作为裁剪算法的部署速度结论。
- 本次交付不包含任何新数值结果或“已通过等价性”的结论；运行后的检查文件才是依据。
