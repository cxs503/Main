author-model: qwen3.7-plus

# 议事会 session-003 · R1 提案（求解器深化视角 · slot: r1-proposal-solver）

> 立场：**10 库谁先上卡深化** + **SDAA 性能/精度基线的可复现性**。
> 纪律：只写本槽位文件；判据全部机器可裁；预算一律 ≤140 调用 / 5400 s；不确定处显式标注 `未验证/不确定`。
> 本文所有数字来自 `artifacts/council_session003/evidence.json`（由 `docs/council/session-003/make_evidence_solver.py` 从工件与现场探测程序化生成，零手抄）。

---

## 0. 一句话结论

**先把「测量」修好，再谈「上卡」。**
TensorLBM 是唯一有 SDAA 设备级证据的库（349 个文件含 `sdaa` 字样、35 个 benchmark 入口、43 个 verified 案例），所以**深化（3D/自由面/多相）第一顺位毫无疑问是 TensorLBM**；但今天**没有任何一条 SDAA 数据能当基线用**——唯一的性能基线工件是 `cv_pass=false`（raw-mean CV=38.81%，5 次连跑墙钟单调递增 2.60×），而我今天在 card25 上现测的 200 次 matmul 也复现了同一现象：**raw-mean CV=11.31%，但 block-median CV=1.51%**。也就是说：**判决结果由统计量选择决定，而不由机器决定**——这正是 PB-005「门无法失败」的变种。
因此本提案的第一优先级不是"多上几张卡"，而是**把性能基线协议变成可复现、可失败、可回读的资产（方向 A）**；在此之上，用**一张现场矩阵 + 单点 PASS** 把"哪个库先上卡深化"钉死（方向 B）；并把两个已经在案发红的精度债（自由面质量守恒 4.48%/1e4、DEM-FEM Hertz 指数 1.41/刚度 44.1%）作为**深化的入场券**（方向 C/D）——**自由面不守恒，谈 3D 上卡只是把错误算得更快。**

---

## 1. 事实基线（全部今日实测或今日复核，可点回工件）

### 1.1 SDAA 性能面：基线目前不成立，且失败模式是"系统性劣化"

| 证据 | 数字 | 判读 |
|---|---|---|
| `m397/artifacts/sdaa_perf_baseline.json`（TensorLBM cavity_3d_full 64³, D3Q19-TRT, 5000 step, 5 重复） | 墙钟 140.403 → 198.783 → 217.529 → 351.581 → 364.672 s；CV=38.81%；MLUPS 9.34 → 3.59；`cv_pass=false` | **单调递增**，末次/首次 = **2.60×**；不是随机噪声，是逐次劣化（热/共租/泄漏三者之一） |
| 口径说明 | `m397` 工件的 `cv_pct=38.81` 用**样本标准差 (n−1)**；若用总体标准差则为 34.71% —— **两种口径都远超 2% 门**，判决不受口径影响 |
| `m621/artifacts/bench.json`（1200² fp32 matmul, `/dev/tcaicard4`, 100 iter） | mean 6.944 ms, **CV=1.34%** | 微基准级可复现 → 机器本身不是坏的；**问题出在长任务与共租** |
| 本日现场探测（`artifacts/council_session003/live_matmul_probe_card25.json`，card25，200 iter） | mean 7.574 / median 7.266 ms；min 6.948 / max 12.29；**raw-mean CV=11.31%**；4 段 block median = 7.466/7.223/7.239/7.186 → **block-median CV=1.51%**；>2×median 尖峰数 **0** | **同一份数据，两种统计量：11.31%(FAIL) vs 1.51%(PASS)**。均值被尾部污染约 4.2%，中位数稳健。**统计量=判决自由度** |
| `m621/artifacts/cross_card_{3,4}card.json`（m10_couette_3d，500 step） | 单卡 mean_step_ms：可见 3 卡时 15.748、可见 4 卡时 16.714 → **+6.13%**；而 `rho/ux/f_sha256` 3 卡与 4 卡**逐位相同** | 正确性可 bitwise 复现（域分解是确定性的）；**性能随"可见卡数"漂移 6%** → 环境变量都能进判据 |
| `teco-smi`（`/opt/tecoai/bin/teco-smi`, TECO-SMI 1.15.0） | 存在；给出每卡 SPE-Util / 温度 / 功耗 / 显存；本次读到 **8 个 index 条目**，`/dev` 下 **32 个 `tcaicard*` 节点**；快照时 index 0 有占用（18%–32% 波动、600 MB） | **修正 session-002 的前提**：当时判据写"SDAA 上无 nvidia-smi 等价物"→ 实际**有 `teco-smi`**，quiet-gate 可机器判。但 **index↔tcaicardN 映射未证实**（8 vs 32），必须显式验证 |

### 1.2 10 库张量后端真实性（现场 grep `/root/tensor-ecosystem`，非引用旧普查）

| 库 | py 文件 | 代码行 | import torch 文件 | torch 占比 | `sdaa` 引用文件 | benchmark 入口 | 判定 |
|---|---|---|---|---|---|---|---|
| TensorLBM | 1965 | 595150 | 1369 | **69.67%** | **349** | 35 | torch-native + 唯一有 SDAA 痕迹 |
| TensorSPH | 23 | 5266 | 18 | 78.26% | 0 | 6 | torch-native（体量最小） |
| TensorDG | 39 | 4684 | 26 | 66.67% | 0 | **18** | torch-native（解析解最密） |
| TensorDEM | 115 | 15893 | 72 | 62.61% | 0 | 4 | torch-native |
| TensorFEM | 492 | 46448 | 294 | 59.76% | 0 | 3 | torch-native |
| TensorFVM | 437 | 45576 | 112 | 25.63% | 0 | 0 | torch 文件占比低（numpy/scipy.sparse 文件 31）→ 核心疑似 CPU |
| TensorPhysics | 102 | 19064 | 12 | 11.76% | 0 | 1 | 偏 CPU |
| TensorCAD | 58 | 20986 | 5 | 8.62% | 0 | 0 | 几何/参数化，不在卡池 |
| TensorPlatform | 4559 | 1718001 | 30 | 0.66% | 12 | 0 | 控制面（不进卡池） |
| TensorBEM | 75 | 18219 | **0** | **0%** | 0 | 0 | **numpy-only（pyproject 只声明 numpy>=1.24）** |

> **诚实边界**：`torch_file_ratio` 只证明"文件里有 `import torch`"，**不证明计算核心跑在张量上**。特别是 TensorFEM 494 个文件里 294 个引 torch，但矩阵装配是否落到 `scipy.sparse` 未验证；TensorLBM 之外 9 库的 `sdaa` 引用**全部为 0**。所以方向 B 必须有"设备探针 + `SHELL_ONLY` 门"，不能只靠这张表下结论。

### 1.3 精度面：深化必须先偿还的两笔债（已在案的 honest-FAIL）

| 债项 | 实测 | 目标 | 超阈倍数 | 来源 |
|---|---|---|---|---|
| Hertz 接触指数 | 1.4108 | 1.5 ± 0.05 | 偏离 0.089（>tol 0.05） | `m404/artifacts/dem_fem_hertz_cpu.json` |
| Hertz 接触刚度误差 | 44.09% | ≤3% | **14.7×** | 同上 |
| 接触能量漂移 | 58.61% | ≤1% | **58.6×** | 同上 |
| 网格加密刚度变化 | 8.43% | <5% | 1.69× | 同上 |
| 自由面质量漂移（dambreak3d） | 4.476 %/1e4 步 | ≤1 %/1e4 | 4.48× | `m407/artifacts/sdaa_dambreak3d.json`（`status=HONEST_FAIL`） |

`m404` 自己的诚实标注是："TensorFEM 在当次工作区不可用，用 DEM3D bonded-sphere 充当半空间"——而今天 `/root/tensor-ecosystem/TensorFEM` **存在**（492 py / 294 引 torch）。**这是"基准本身错了"，不是"物理难"**：晶格弹簧组装不可能复现连续体 Hertz。→ 方向 D 的第一件事是"用真 FEM 重跑"，而不是调参。

---

## 2. 判决：谁先上卡深化（结论先行）

| 排位 | 库 | 深化方向 | 为什么是它 / 不是它 | 证据强度 |
|---|---|---|---|---|
| **①** | **TensorLBM** | **3D 湍流/自由面/多相**（cavity_3d、dambreak3d、相场 Laplace、sphere Re=100） | 唯一有 349 文件 SDAA 痕迹 + 35 benchmark + 43 verified；已有 bitwise 可复现的多卡域分解（m621）；同时**两笔精度债都落在它身上**，深化=还债 | REAL（但基线不可用） |
| ② | TensorDEM | 颗粒侧上卡（接触搜索/邻居表） | 62.6% torch、115 文件 15.9k 行，体量小、上卡风险低；是 DEM-FEM 耦合的**可行的一半** | MOCK 级：无 sdaa 痕迹、无设备证据 |
| ③ | TensorDG | 收缩阶基线（18 个 benchmark + 解析解密集，最易做"能失败的门"） | 66.7% torch、4.7k 行最小；作为**"非 LBM 库能否上卡"的判别器**，一次跑通即证明卡池不只服务 LBM | WITHHELD |
| ④ | TensorFEM | 耦合的 FEM 侧（真 FEM 重做 Hertz） | 59.8% torch 但 46k 行、装配可能走 scipy；**先 CPU 验物理，再谈上卡** | WITHHELD |
| ⑤ | TensorSPH | 液滴入水/自由面（与 LBM 交叉验证） | torch 占比最高(78%)但只有 5.3k 行 6 benchmark；且 owner P0 已点名 SPH 独立误差指标 | WITHHELD |
| — | TensorFVM | **暂不上卡** | numpy 343 文件 vs torch 112 → 计算核心疑在 CPU；上卡前需先做"核心张量化"立项 | — |
| — | TensorBEM | **不能直接上卡** | torch 文件 **0**、`pyproject` 只依赖 numpy → 上卡只能是外壳。**但它是我们冲 RAO 精度的引擎**，正确立项是「numpy→torch 移植 1 个 kernel 族」，不是"上卡" | NOT_SUPPORTED |
| — | TensorPhysics / TensorCAD / TensorPlatform | 不进卡池 | 前者偏 CPU 经验模型，中者几何，后者控制面 | — |

**结论一句话**：**深化= TensorLBM；"能不能别只靠 LBM"的判别 = TensorDG 单点；耦合可行性的两半 = DEM（立即可）+ FEM（先 CPU 验物理）；BEM 要先移植后上卡，别假装。**

---

## 3. 候选方向（4 个，全部"今天可入池"）

### 方向 A（P0，**建议今日首单**）SDAA 性能基线可复现化：把判决权从"统计量"手里收回来

**① 目标**：产出一份**协议 + 一张安静窗实测基线 + 一个可回读端点**，使 SDAA 性能数字同机同配置重复可得（重复间 CV ≤2%），并能在门失败时明确报 `BASELINE_INVALID` 而不是"挑一个好看的统计量"。

**② mission 拆解**

| ID | repo | 交付物 | 机器判据（全部脚本可裁） | 预算 | 卡 |
|---|---|---|---|---|---|
| **A1** 噪声特征化与统计量裁决 | TensorLBM（脚本落 `scripts/`）+ TensorPlatform（文档） | `artifacts/perf_noise_char.json` + `docs/sdaa_baseline/PERF_NOISE.md` + `scripts/perf_noise_char.py` | ① ≥3 个独立来源：`m397` 5 连跑工件、`m621` matmul 记录、现场 probe ≥200 iter；json 内每条带 `path`+`sha256`；② 对每来源同时算 raw-mean CV / block-median CV / p10 三统计量；③ 必须给出 ≥1 个**同一数据、不同统计量、判决相反**的实例（本日实测：11.31% FAIL vs 1.51% PASS），并在 md 中给出推荐统计量及理由；④ md 内数字全部由 json 渲染：`grep` 校验 md 出现的数值集合 ⊆ json 数值集合，手抄计数=0；⑤ 产出「quiet-gate 规格」一节，写死 `teco-smi` 字段名与阈值 | 140/5400 s（纯离线，预计 <15 min） | **0** |
| **A2** 安静窗协议实测（含 index 映射探测） | TensorLBM | `artifacts/sdaa_perf_baseline_v2.json` + `docs/sdaa_baseline/PERF_PROTOCOL.md` | ① **映射探测**：在目标卡 alloc 512 MB 张量，抓 `teco-smi` 前后差，得 `tcaicardN ↔ index` 映射（映射不确定则记 `mapping_unknown=true` 并降级为粗判）；② 每次重复前 `teco-smi` 快照须满足 `SPE-Util=0%` 且 `Mem<100MB`；跑后若出现非自身占用 → 该重复 `contended:true` 作废；③ **9 次**重复，每次 ≥100 step warmup + 200 次计时；④ 重复间 **p50 CV ≤2%**；⑤ 同窗口控制微基准（1200² matmul）**CV ≤2%**；⑥ 漂移门：末次 p50 / 首次 p50 ∈ [0.95, 1.05] **且** Mann-Kendall 趋势检验 p>0.05；⑦ 输出 MLUPS(p50 口径)、显存峰值、同配置 CPU 参考与加速比；⑧ 任一不满足 → 产物首行 `BASELINE_INVALID` + 逐条未过原因，**禁止改判据或删除失败重复** | 140/5400 s（独占 1 卡 ≤45 min） | **1（独占，禁共租）** |
| **A3** 基线回读端点（平台侧闭合） | TensorPlatform | `GET /calculations/sdaa_baseline` + `schemas/sdaa_perf_baseline_v2.schema.json` + pytest | ① jsonschema 校验通过；② 端点数值与工件 json **逐字段 diff = 0**（脚本断言，非人眼）；③ pytest ≥3 例：200、schema valid、工件缺失→503 且 `reason` 非空；④ 报告自动汇编 1 份，数字全部来自 json | 140/5400 s | **0** |

**③ 卡位分配**：A1 0 卡；A2 **1 卡独占 ≤45 min**；A3 0 卡。峰值 **1 卡**。

**④ 风险与依赖**
- **R-A1（已实测的坑）**：节点有共租噪声（raw CV 11.31%），`teco-smi` 只暴露 8 个 index 对 32 个 `/dev` 节点 → 映射必须先在 A2 里验证，否则 quiet-gate 是假的。
- **R-A2**：`triton 3.4.0` 无 SDAA 后端（`m401` 普查实证 `backend_dirs=['amd']`），`teco_inductor` 不完整 → 任何 `torch.compile` 路径的结果**不得**记为基线；A2 必须写 `compile_route: eager|inductor`。
- **R-A3**：预算内 9 次重复可能超时 → 先用 A1 的耗时估算把 case 降到 `nx=48³ / 500 step`，并在产物里写 py 与 step 数（不同配置的 MLUPS 不可横向比）。
- **依赖**：A3 依赖 A1/A2 的产物；A2 依赖 A1 的协议（若 A1 未过，A2 用本文 §1.1 的临时协议）。

**⑤ 放弃条件**
- A2：3 次尝试都拿不到安静窗（mid-run 出现非自身占用），或**控制微基准自身 CV >5%** → 停止刷时间，只交《node099 共租噪声报告》并把 `BASELINE_INVALID` 写在首行，转入"申请独占窗口"流程。
- A1：若 3 个来源的 `sha256` 有对不上的（工件被覆写）→ 只用存活的来源，并在 json 里记 `sources_dropped[]`，不得混用不同配置的数字算同一个 CV。

---

### 方向 B（P0-P1）选型矩阵 → 设备探针 → 单点 PASS：把"哪个库先上卡"变成可失败的门

**① 目标**：用三单把"10 库谁先上卡深化"从表格争论变成本波内**至少 1 个非 LBM 库在 SDAA 上跑出与 CPU 一致的解**（或显式证伪：`SHELL_ONLY`）。

**② mission 拆解**

| ID | repo | 交付物 | 机器判据 | 预算 | 卡 |
|---|---|---|---|---|---|
| **B1** 张量后端真实性矩阵（本文 §1.2 的脚本化复现） | TensorPlatform（落 `docs/inventory/` 作 INV 台账补全） | `artifacts/tensor_backend_matrix.json` + `scripts/tensor_backend_matrix.py` | ① 10 库各一行，字段含 `py_files/loc/torch_files/sdaa_files/cpu_native_files/benchmark_entries/compute_backend/evidence_file_lines[]`；② `compute_backend` 规则写在脚本里且**输入相同则输出 sha256 稳定**（跑 2 次比对 sha256 相等）；③ 每库 ≥2 条 `file:line` 证据行（脚本抓 import 原文，**零手抄**）；④ 必须复现本文 §1.2 的 10 行数字，差异须逐项解释（差=0 才算过） | 140/5400 s | **0** |
| **B2** 设备探针 + `SHELL_ONLY` 门（按 B1 排序取前 2 个非 LBM 库，各 1 单或合并 1 单串行） | TensorDEM、TensorDG（首选） | `artifacts/sdaa_probe_<lib>.json` + `docs/sdaa_compat/<LIB>.md` | ① 打印 `.device` 为 `sdaa:0` **且该张量参与 ≥1 次完整 step/装配**；② 同最小算例 sdaa vs CPU **逐值 L2 ≤1e-6（fp32）**；③ **假加速门**：若实测加速比 <1%（timediff 噪声级）或 profiler 显示计算落在 CPU → 判 `SHELL_ONLY`（这就是 R-B2 的可失败形式）；④ `float64` 支持性显式记录（matmul 正确性 + 相对误差 ≤1e-6 → 支持，否则记精度天花板） | 每单 140/5400 s（≤2 卡串行） | **1/单，最多同时 2** |
| **B3** 单点物理 PASS（谁是第二个上卡的库，只认一次物理门） | 胜者：TensorDG（线性对流收敛阶）或 TensorDEM（单颗粒 Hertz 解析） | `artifacts/sdaa_firstpass_<lib>.json` + `docs/sdaa_compat/FIRSTPASS.md` | ① DG：数值收敛阶与理论阶差 **≤0.15**，且解析解 L2 ≤1e-3；DEM：单颗粒 Hertz 指数 **1.5±0.05**、刚度误差 **≤3%**；② sdaa vs CPU 同算例 L2 ≤1e-6；③ 记录 `compile_route`；④ 与 CPU 参考值同表并列（禁止只报 sdaa 一个数） | 140/5400 s | **1** |

**③ 卡位分配**：B1 0 卡；B2 2 卡（两个库并行，各独占，互不 OOM）；B3 1 卡（等 B2 胜者）。峰值 **2 卡**。

**④ 风险与依赖**
- **R-B1**：TensorDEM/DG 可能"import torch 但 loop 在 python 里逐粒子跑" → 加速比≈1，正是 `SHELL_ONLY` 门要抓的；这**不算失败**，是有效结论（等于说"上卡预算应该给别处"）。
- **R-B2**：`float64` 在 SDAA 上支持未知；解析解验证常需 double → 冲突时用 fp32 并把"精度天花板"写进 md。
- **R-B3**：TensorDG 的 18 个 benchmark 可能依赖 CPU 侧 numpy 网格工具 → 探针要区分"计算在卡上、准备在 CPU"，记 `host_share_pct`（若可测）。
- **依赖**：B2/B3 依赖 B1 排序；B3 若胜者为 DEM，则与方向 D 的 Hertz 判据同源（可复用，但**必须用不同算例**，避免"同一错误自洽验证"）。

**⑤ 放弃条件**
- B2：若前 2 个库都判 `SHELL_ONLY` 或 import 即崩 → 关 B3，把结论写成"10 库中仅 TensorLBM 具备真实卡上计算能力（其余均需先做核心张量化）"，**不得用 LBM 单例代表整个生态**。
- B3：若胜者库在 CPU 上连解析解都过不了（物理本身就错）→ 停下上卡，转"该库物理正确性立项"，并如实标注 `PHYSICS_BLOCKED`。

---

### 方向 C（P1）自由面/多相：先修 CPU 守恒，再证上卡等价

**① 目标**：把 `m407` 的 4.48 %/1e4 质量漂移压到 ≤0.1 %/1e4，然后证明"SDAA 上跑的与 CPU 逐位等价"，这样 3D 自由面的上卡才是**加速**而不是**加速错误**。

**② mission 拆解**

| ID | repo | 交付物 | 机器判据 | 预算 | 卡 |
|---|---|---|---|---|---|
| **C1** dambreak3d 质量守恒根因 | TensorLBM | `artifacts/dambreak_mass_rootcause.json` + `docs/sdaa_triage/FREESURFACE.md` | ① 复现 drift（4.476 %/1e4，容差 ±20%）；② 定位到函数级：≥1 个 `file:line` 候选缺陷 + ≥1 个**可失败的最小实验**（关掉该分支 → drift 变化 ≥3×）；③ 修复后两个网格（40×20×16 与 80×40×32）drift 均 **≤0.1 %/1e4**；④ 若未达标 → `HONEST_FAIL` + 给出 2D↔3D 一致性反例（同算法 2D 是否守恒）；⑤ 质量时序数据落 artifacts（含 `rho_gas=1.0` 的稳定性约束，`rho_gas=0.1` 触发 `TopologyTransactionError` 的事实必须保留） | 140/5400 s | **0** |
| **C2** 上卡等价性 | TensorLBM | `artifacts/sdaa_dambreak3d_equiv.json` | ① 同配置 100 step，sdaa vs CPU 的 `rho/ux/质量` **sha256 全等**（`m621` 已证域分解可 bitwise 等）；不等则给出**首个分歧 step** 与量级差；② 若只到 L2 ≤1e-6 也接受，但必须写"非 bitwise"及原因；③ 显式记录 `compile_route` 与 triton 缺席事实；④ 注意 `rho_gas=1.0` 约束（不得为凑等价改物理参数） | 140/5400 s | **1** |
| **C3** 多相 Laplace 定律（精度基线） | TensorLBM | `artifacts/laplace_law.json` + `docs/sdaa_triage/LAPLACE.md` | ① `Δp·R/σ` vs 理论值（2D=1、3D=2）相对误差 **≤3%**；② 两网格 + 加密误差降幅符合阶数（Richardson/GCI 之一，须写出估阶）；③ 与历史 m24/m25/m52 数值逐项对照表（差异>10% 必须解释） | 140/5400 s | **0** |

**③ 卡位分配**：C1 0 卡（CPU）；C2 1 卡（≤30 min）；C3 0 卡。峰值 **1 卡**，且与 A2 错峰（都要求安静窗）。

**④ 风险与依赖**
- **R-C1**：根本原因可能是"free-surface 的 fill-fraction 阈值/界面重建"设计性缺陷 → 预算内可能只交根因不交修复；那也要如实 `HONEST_FAIL`，**不许把阈值调松让数字好看**（PB-005）。
- **R-C2**：等价性判据依赖 C1（守恒没修，等价性只是"同样错"）→ **C2 必须在 C1 之后**；若 C1 FAIL，C2 只报"等价性"不报"精度"。
- **R-C3**：相场/伪势的 σ 依赖界面厚度；若界面 <3 格，`Δp·R/σ` 无法对齐 → 记 `WITHHELD` + 交界面分辨率扫描的最小实验设计。

**⑤ 放弃条件**
- C1：预算内既无根因又无可失败实验 → 关单，交《自由面重写立项建议》，不再做调参。
- C2：若 sdaa 路径在 100 step 内触发 `TopologyTransactionError` 或 OOM → 记 `BLOCKED`（环境），不许降网格后宣称等价。
- C3：若两网格误差都在 20% 以上且不随加密改善 → 判方法性失败（`NOT_SUPPORTED` 级），交回算法选型。

---

### 方向 D（P1，可后置）DEM-FEM 耦合：把"基准错了"和"物理错了"分开

**① 目标**：用**真正的 TensorFEM**（`/root/tensor-ecosystem/TensorFEM`）重做 Hertz 接触（或退化为梁挠度+单颗粒解析双线），产出一份**能失败**的耦合基准，替代 `m404` 那次"用 DEM 晶格冒充连续体"的判据。

**② mission 拆解**

| ID | repo | 交付物 | 机器判据 | 预算 | 卡 |
|---|---|---|---|---|---|
| **D1** 真 FEM Hertz（或双线降级） | TensorFEM + TensorDEM | `artifacts/dem_fem_hertz_v2.json` + `docs/sdaa_coupling/HERTZ.md` | ① 记录 TensorFEM `git sha` + 依赖安装证据（`pip show torch` 版本、import 成功的行）；② 若 FEM 无接触/罚函数单元 → 降级为「梁集中力挠度 ε≤3%（vs Euler-Bernoulli 解析）」+「DEM 单颗粒 Hertz 指数 1.5±0.05、刚度 ≤3%」**两条独立线**，并显式禁止用 bonded-sphere 冒充连续体（引用 `m404` 的 44.09% 作为反例）；③ 4 条门：指数 1.5±0.05 / 刚度 ≤3% / 能量漂移 ≤1% / 网格加密刚度变化 <5%；④ 与 `m404` 的 1.4108 / 44.09% / 58.61% / 8.43% 逐项对照表 | 140/5400 s | **0（CPU 优先）** |
| **D2** 耦合接口 schema 冻结 + 单步守恒 | TensorFEM + TensorDEM | `schemas/dem_fem_coupling_v1.json` + `artifacts/dem_fem_step1.json` | ① schema 自校验通过（jsonschema）+ 两库往返序列化**位等**（dtype/device 字段显式）；② 单步能量交换守恒误差 ≤1e-3；③ 冻结后再谈上卡（本波不上卡） | 140/5400 s | **0** |

**③ 卡位分配**：D1、D2 均 0 卡（CPU）。若 D1 的梁算例在 CPU 上单次 <5 min，可留 1 卡给 D 的后续波次（不在本波）。

**④ 风险与依赖**
- **R-D1**：`m404` 记录"TensorFEM 当次工作区不可用"与今天仓库存在**冲突** → D1 第一步必须裁决这个矛盾（是 clone 缺失还是依赖装不上），并在产物写清；否则会被当成"又一次环境运气"。
- **R-D2**：DEM 接触时间步 dt≈1.69e-5 与 FEM 隐式步长差 3 个量级 → 单步守恒门可能因时间积分不一致而假失败，应允许子循环，但**必须写清子循环比例**。
- **依赖**：D2 依赖 D1 的物理门先立；两者都不依赖卡，可与 A/B 并行不抢资源。

**⑤ 放弃条件**
- D1：若 TensorFEM 在 5400 s 内装不起来（依赖缺失）→ 记 `ENV_BLOCKED`（**不是**物理 FAIL），只交梁挠度单线，并给出环境修复的最小清单。
- D2：若两库 dtype 约定不一致（fp32 vs fp64）导致位等不可达 → 记 `WITHHELD` + 给出统一 dtype 的迁移建议。

---

## 4. 推荐波次与入池顺序（今天就能派）

| 序 | mission | 卡 | 预算 | 为什么先它 |
|---|---|---|---|---|
| 1 | **A1** 噪声特征化与统计量裁决 | 0 | 140/5400 | 纯离线，`<15 min`，立刻把"判决自由度"这件事钉死；无依赖 |
| 2 | **B1** 张量后端矩阵脚本化 | 0 | 140/5400 | 0 卡，今日可跑；给 B2/B3 排序 |
| 3 | **A2** 安静窗协议实测（含 index 映射） | 1 独占 | 140/5400 | 需要 A1 的协议；**本波唯一"必须独占卡"的单** |
| 4 | **C1** dambreak3d 质量守恒根因 | 0 | 140/5400 | 0 卡、与卡池完全并行；深化入场券 |
| 5 | **D1** 真 FEM Hertz / 双线降级 | 0 | 140/5400 | 0 卡；同时裁决 `m404` 的矛盾 |
| 6 | **B2** 设备探针（2 库） | 2 串行/并行 | 各 140/5400 | 依赖 B1 排序；A2 之后跑，避免共租污染 |
| 7 | **C2 / C3 / B3 / D2** | 1 / 0 / 1 / 0 | 各 140/5400 | 依 A2/B2/C1/D1 结果放行（有条件入池） |

**卡位总账**：本波**峰值并发 2 卡**，卡时 ≤ 4 卡·小时（A2 ≤0.75 h + B2 2×0.5 h + C2 ≤0.5 h + B3 ≤0.5 h）。
**为什么不是 8 卡 16 卡**：现场实测共租噪声 raw-mean CV 11.31%（§1.1），并发越高，A2 的 CV ≤2% 门越必然失败——**先把 1 卡量准，再谈多卡扩展**；`m621` 已证多卡正确性 bitwise 可复现，所以"多卡"不缺正确性证据，缺的是**性能口径**。

---

## 5. KPI 对账（本波提案对各 KPI 的实际贡献）

| KPI（guide_tracker 缺口） | 本提案贡献 | 是否闭合 |
|---|---|---|
| 计时基线（现状有 p50/p90/p99，但 SDAA 侧基线 `cv_pass=false`） | A1+A2+A3：给出 SDAA 侧**可复现**基线 + 回读端点 | 从 0 → 1（有基线），**前提是 A2 过门** |
| 对标算例 ≥10（现 1 完成 + 2 在飞） | C3（相场 Laplace vs 理论）、D1（Hertz/梁挠度解析）各贡献 1 个**独立参考** | 2 个新独立参考（不计入既有 3 个） |
| 精度实测 | C1（自由面守恒）、C3（Laplace ≤3%）、D1（Hertz 4 门）三处**如实数据** | 提供"实测"而非"承诺" |
| 门禁覆盖 100% / 已合并功能复核通过率 | A1 的"统计量判决自由度"判例 + A2 的 `BASELINE_INVALID` 纪律 = 让门**能失败** | 新增 1 条判例候选（PB 库） |
| 工具接入 ≥5 类（现 4/5） | B2 的成功上卡库即第 5 类工具的证据（非 LBM） | 待 B2 结果 |
| 千工况/万样本 | 本提案**不覆盖**（属 software 槽位） | 0 |

---

## 6. 诚实标注：不确定、与既有前提的冲突

1. **`m397` 的 5 次连跑单调劣化（2.60×）根因未定**：热降频 / 共租 / 内存泄漏 / 长任务调度排队，四种可能我**没有区分证据**。A2 的 quiet-gate + 控制微基准是把这四者分开的最小实验设计，**不是已解决问题**。
2. **`teco-smi` 的 8 index vs 32 `/dev` 节点映射未证实**：可能是"1 index = 4 个 tcaicard 逻辑节点"或采集侧截断，我**未验证**。A2 必须先做映射探测，否则 quiet-gate 会指错卡。
3. **session-002 一处前提被证伪**：当时判据写"记录 nvidia-smi 等价物不可用事实（SDAA 上无 nvidia-smi）"——实际 `teco-smi 1.15.0` 存在并给出 SPE-Util/温度/功耗/显存。建议后续 mission 统一改用 `teco-smi`。
4. **`torch_file_ratio` 不等于"算在卡上"**：本文 §1.2 只是"有没有 import torch"，TensorFVM(25.6%)/TensorPhysics(11.8%) 的真实计算位置**未验证**；TensorFEM 的矩阵装配是否走 `scipy.sparse` **未验证**（B2 的 `SHELL_ONLY` 门就是为了抓这个）。
5. **`m404` 与仓库存在性矛盾**：`m404` 记"TensorFEM 在当次工作区不可用"，而今天 `/root/tensor-ecosystem/TensorFEM`（492 py / 294 引 torch）存在。二者必有一方是"当时环境问题"。D1 必须先裁决，**在此之前不得再引用 `m404` 的 44.09% 当作 TensorFEM 的能力证据**。
6. **`float64` 在 SDAA 上的支持性未知**：解析解验证常需 double；若 SDAA 不支持，则所有"vs 解析解 ≤3%"的判据都会撞精度天花板，需改用 fp32 并把容差与物理误差分开声明。
7. **本提案的范畴外事项**：千工况/万样本（software 槽位）、冰区/推进能力（平台当前 NOT_SUPPORTED）、`triton SDAA 后端修复`（需上游，非本波预算内可解决）——本提案不承诺。
8. **预算帽未上调**：本提案全部单 ≤140 调用 / 5400 s，未假设 session-002 提议的 200/7200。

---

## 7. 与 session-002 的关系（不重复劳动）

| session-002 已做 | 结论 | 本提案接续方式 |
|---|---|---|
| M2 单卡性能基线（5 次重复） | `cv_pass=false`（38.81%） | **A1/A2**：不改判据刷 PASS，而是换协议（安静窗 + median + 控制微基准 + 漂移门）并允许 `BASELINE_INVALID` |
| M6 三库普查（LBM/FEM/BEM） | PARTIAL/BLOCKED/BLOCKED | **B1** 扩到 10 库并脚本化；**B2** 补"设备探针 + `SHELL_ONLY`"这一层（普查只是 grep，不证算在卡上） |
| M9 DEM-FEM Hertz CPU 标杆 | honest-FAIL（指数 1.41、刚度 44.1%） | **D1** 用真 TensorFEM 重做或双线降级 —— 先修"基准错了" |
| M12 自由面 3D 守恒 | honest-FAIL（4.48 %/1e4） | **C1** 根因 → **C2** 上卡等价 → **C3** 多相精度 |
| M8 3D 球体阻力 | （未在本提案展开） | 并入 C 线作为 3D 湍流精度参考，不与 B3 复用同算例 |

**建议议事会裁决点（3 条）**
1. 是否接受"**本波最多 2 卡并发**"（理由：实测共租噪声 raw CV 11.31%，并发会污染唯一的性能门）？
2. 是否同意**先修自由面守恒（C1）再谈 3D 自由面上卡**——即 C2 有条件入池？
3. 是否同意把 `teco-smi` quiet-gate 写进所有后续 SDAA 性能类 mission 的通用判据（替换已证伪的"无 nvidia-smi"前提）？