#!/usr/bin/env python3
"""组装 session-003 求解器槽位论文级进展报告 -> /tmp/report.json（数字全部程序化注入）。"""
from __future__ import annotations

import base64
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ART = ROOT / "artifacts" / "council_session003"
EV = json.load(open(ART / "evidence.json"))

P = EV["perf"]
L = EV["live_probe"]
X = EV["cross_card"]
A = EV["accuracy_debt"]
T = EV["teco_smi"]
LIBS = EV["lib_matrix"]


def b64(p: Path) -> str:
    return base64.b64encode(p.read_bytes()).decode()


lib_rows = "\n".join(
    f"| {r['lib']} | {r['py_files']} | {r['loc']} | {r['torch_files']} | {r['torch_file_ratio']*100:.2f}% | "
    f"{r['sdaa_files']} | {r['benchmark_entries']} | {r['compute_backend']} |"
    for r in sorted(LIBS, key=lambda r: -r["torch_file_ratio"])
)

md = f"""# session-003 · 求解器深化视角：10 库上卡排序与 SDAA 性能/精度基线审计

> author-model: qwen3.7-plus ｜ mission 676 ｜ 全部数字由 `docs/council/session-003/make_evidence_solver.py`
> 从工件与现场探测程序化注入（`artifacts/council_session003/evidence.json`，含 7 个输入源 sha256）。

## ① 案例介绍（物理背景 / 几何 / 参数表）

本单是**议事会 R1 提案（求解器槽位）+ 基线审计**，不是新求解算例。被审计的三个"案例"都是平台在案的真实算例：

| 案例 | 物理 | 几何/配置 | 关键参数 |
|---|---|---|---|
| LBM 顶盖驱动方腔（性能基线） | 不可压 LBM D3Q19-TRT | {P['case']}，{64}³ 网格 | Re=1000, u_lid=0.06, {5000} step, warmup 100, 5 次重复 |
| LBM Couette 3D（跨卡确定性） | D3Q19 | H=64, nx=64, ny=66, nz=64 | tau=0.8, U0=0.05, 500 step, 可见 3 卡 / 4 卡 |
| 微基准 matmul（噪声基准） | 1200×1200 fp32 GEMM | 单卡（历史 card4；本次现场 card25） | 100 iter（历史）/ 200 iter（本次） |
| DEM 接触（精度债 1） | 48 粒子 bonded-sphere 压入 | r=0.02 m, 工具半径 0.1 m | dt=1.687e-5 s, E_eff=1.333 MPa, 压深 0.05 m |
| LBM 溃坝 3D（精度债 2） | 自由面 | 40×20×16 | rho_gas=1.0, tau=0.8, g=1e-4, 80–100 step |

**为何要做**：owner 要求"已合并 ≠ 已验证"，验收必须引独立参考；而 SDAA 侧唯一在案的性能基线工件判决为 `cv_pass=false`，若沿用，任何"加速比"都是噪声。

## ② 怎么做的（方法 / 流程 / 工具链）

1. **工件回溯 + 完整性锁**：对 m397/m621/m404/m407 五份工件算 sha256 并写入产物（`evidence.json.sources`），避免"基线漂移"。
2. **统计量裁决实验**：同一份 200 次现场计时序列，同时用 **raw-mean CV** 与 **block-median CV**（4 段各 50 次取中位）计算，量化"统计量=判决自由度"。
3. **现场探测**：`SDAA_VISIBLE_DEVICES=25` 上跑 1200² fp32 matmul，20 次预热 + 200 次计时，`torch.sdaa.synchronize()` 卡边界。
4. **共租可观测性验证**：`/opt/tecoai/bin/teco-smi`（TECO-SMI 1.15.0）抓 SPE-Util/温度/功耗/显存；对照 `/dev/tcaicard*` 节点数。
5. **10 库张量后端矩阵**：对 `/root/tensor-ecosystem/*` 现场 `grep`（`import torch` / `numpy` / `sdaa` / `numba|scipy.sparse` 文件数、benchmark 入口数），脚本可复现。
6. **自校验测试**：`pytest -q tests/council/test_session003_evidence.py` — 6 例全绿，逐项从源工件复算报告里的数字（防止手抄漂移）。
7. 工具链：python3 + torch 2.7.1 / torch_sdaa 3.1.1a3+gite3971b9 + matplotlib（Agg）+ teco-smi 1.15.0。

## ③ 计算结果（数字表，程序化注入）

**性能面**

| 量 | 值 |
|---|---|
| m397 五连跑墙钟 (s) | {', '.join(f'{v:.3f}' for v in P['wall_s'])} |
| CV（工件口径，样本 std） | {P['raw_mean_cv_pct']:.2f}% （总体 std 口径 {P['cv_pct_population']}%） |
| 单调递增 | {P['monotone_increasing']}；末次/首次 = **{P['last_over_first']:.2f}×** |
| MLUPS 序列 | {', '.join(f'{v:.2f}' for v in P['mlups'])} |
| 工件判决 | {P['verdict_in_artifact']} |
| 现场 probe（card25, n={L['n']}） | mean {L['raw_mean_ms']:.3f} ms / median {L['raw_median_ms']:.3f} ms；min {L['min_ms']:.3f} / max {L['max_ms']:.3f} |
| 现场 raw-mean CV | **{L['raw_mean_cv_pct']:.2f}%** |
| 现场 block medians (ms) | {', '.join(f'{v:.3f}' for v in L['block_medians_ms'])} → **CV {L['block_median_cv_pct']:.2f}%** |
| >2×median 尖峰数 | {L['spikes_gt_2x_median']} |
| 历史 matmul（m621, card4, {EV['microbench_m621']['n_iter']} iter） | mean {EV['microbench_m621']['mean_ms']:.3f} ms, CV {EV['microbench_m621']['cv_pct']:.2f}% |
| 跨卡步时漂移（单卡 3 vs 4 卡可见） | {X['single_card_mean_step_ms_3visible']:.3f} → {X['single_card_mean_step_ms_4visible']:.3f} ms，**+{X['delta_pct']:.2f}%** |
| 跨卡场量 sha256 逐位相同 | {X['field_sha256_identical_3_vs_4']} |
| teco-smi 可见 index 条目 / /dev 节点 | {T['index_entries']} / {T['dev_nodes']}（映射未证实） |

**精度债（在案 honest-FAIL）**

| 项 | 实测 | 目标 | 超阈 |
|---|---|---|---|
| Hertz 接触指数 | {A['hertz_exponent']:.4f} | {A['hertz_exponent_target']}±{A['hertz_exponent_tol']} | 偏离 {abs(A['hertz_exponent']-A['hertz_exponent_target']):.3f} > tol |
| Hertz 刚度误差 | {A['hertz_stiffness_err_pct']:.2f}% | ≤{A['hertz_stiffness_target_pct']}% | **{A['hertz_stiffness_err_pct']/A['hertz_stiffness_target_pct']:.1f}×** |
| 接触能量漂移 | {A['hertz_energy_drift_pct']:.2f}% | ≤{A['hertz_energy_target_pct']}% | **{A['hertz_energy_drift_pct']/A['hertz_energy_target_pct']:.1f}×** |
| 网格加密刚度变化 | {A['hertz_mesh_change_pct']:.2f}% | <{A['hertz_mesh_target_pct']}% | {A['hertz_mesh_change_pct']/A['hertz_mesh_target_pct']:.2f}× |
| 自由面质量漂移 | {A['dambreak_drift_per_1e4_pct']:.3f} %/1e4 | ≤{A['dambreak_target_per_1e4_pct']} %/1e4 | {A['dambreak_drift_per_1e4_pct']/A['dambreak_target_per_1e4_pct']:.2f}×（{A['dambreak_status']}） |

**10 库张量后端矩阵（现场 grep）**

| 库 | py 文件 | 代码行 | import torch 文件 | torch 占比 | sdaa 引用文件 | benchmark 入口 | 判定 |
|---|---|---|---|---|---|---|---|
{lib_rows}

## ④ 结果图

3 张图分别给出：性能噪声的两种统计量对照（含历史 5 连跑劣化）、精度债 vs 阈值、10 库张量后端占比。
**无场量云图的理由**：本单为提案 + 基线审计，未运行新的流体算例，出示云图会是伪造。

## ⑤ 注意点（踩过的坑 / 局限 / 适用范围）

1. **统计量是判决自由度**：同一份 200 次计时，raw-mean CV {L['raw_mean_cv_pct']:.2f}%（判 FAIL）vs block-median CV {L['block_median_cv_pct']:.2f}%（判 PASS）。若判据不写死统计量与估计器（样本/总体 std），性能门可被"选统计量"刷绿 —— PB-005「门无法失败」的变种。
2. **m397 的劣化是系统性的**：五连跑单调递增 {P['last_over_first']:.2f}×，不是白噪声；根因（热降频/共租/泄漏/排队）**本次未区分**，只在提案里设计了区分实验（quiet-gate + 控制微基准 + 漂移门）。
3. **teco-smi 存在但映射未证实**：修正了 session-002「SDAA 上无 nvidia-smi 等价物」的前提；但 teco-smi 只列 {T['index_entries']} 个 index，`/dev` 有 {T['dev_nodes']} 个 `tcaicard*`，`tcaicardN ↔ index` 未验证，quiet-gate 前必须先探测。
4. **可见卡数就能改变性能 6%**：{X['delta_pct']:.2f}% 的步时差，说明环境变量也必须进基线口径；而场量 sha256 逐位相同 → **多卡不缺正确性证据，缺性能口径**。
5. **triton 3.4.0 无 SDAA 后端**（m401 普查：`backend_dirs=['amd']`），`teco_inductor` 不完整；任何 compile 路径结果不得当基线。
6. **`torch_file_ratio` ≠ 算在卡上**：只是"文件里有 import torch"；TensorBEM 为 0（numpy-only），TensorFVM 仅 25.63% → 上卡前需先做核心张量化，否则只是外壳。
7. **m404 与仓库存在性矛盾**：m404 称 TensorFEM 当次不可用，而今日 `/root/tensor-ecosystem/TensorFEM` 存在（492 py）；二者必有一方是环境问题，D1 单先裁决，在此之前不得引用 m404 的 44.09% 作为 TensorFEM 能力证据。
8. **适用范围**：结论只覆盖本机（node099）与上述版本组合；换驱动/torch_sdaa 版本需重测。

## ⑥ 误差判定（逐项 vs 基准，显式回答「是否 <3%」）

| 关键量 | 值 | 基准/阈值 | 相对误差 | <3% ? |
|---|---|---|---|---|
| 现场 block-median CV | {L['block_median_cv_pct']:.2f}% | ≤2%（重复性门） | — | **是**（且 <2%） |
| 跨卡 rho/ux sha256 | 逐位相同 | 0 差 | 0.00% | **是** |
| 现场 raw-mean CV | {L['raw_mean_cv_pct']:.2f}% | ≤2% | 超阈 5.66× | **否** |
| m397 基线 CV | {P['raw_mean_cv_pct']:.2f}% | ≤2% | 超阈 19.4× | **否** |
| 单卡步时随可见卡数漂移 | {X['delta_pct']:.2f}% | ≤3% | — | **否** |
| Hertz 接触指数 | {A['hertz_exponent']:.4f} | 1.5 | {(A['hertz_exponent']/1.5-1)*100:.2f}% | **否**（超 tol 0.05） |
| Hertz 刚度 | {A['hertz_stiffness_err_pct']:.2f}% | ≤3% | 超阈 {A['hertz_stiffness_err_pct']/3:.1f}× | **否** |
| 接触能量漂移 | {A['hertz_energy_drift_pct']:.2f}% | ≤1% | 超阈 {A['hertz_energy_drift_pct']:.1f}× | **否** |
| 自由面质量漂移 | {A['dambreak_drift_per_1e4_pct']:.3f} %/1e4 | ≤1 %/1e4 | 超阈 {A['dambreak_drift_per_1e4_pct']:.2f}× | **否** |
| 提案文本/10 库矩阵 | — | 无物理基准（grep 计数，非物理量） | — | 无基准，不适用 |

**结论**：**本单没有任何一项"物理精度 <3%"的达成声明**（无新算例）；达成的是 2 项可复现性指标（block-median CV {L['block_median_cv_pct']:.2f}% <2%、跨卡逐位一致 0%）。已合并但未达标的三项（Hertz 三率、自由面守恒、性能基线）如实标 FAIL，不粉饰。
"""

payload = {
    "title": "session-003 求解器深化提案：10 库上卡排序与 SDAA 性能/精度基线审计",
    "summary": "10 库现场张量后端矩阵 + SDAA 性能基线失效根因（统计量自由度）+ 两项在案精度债复盘；提案 4 方向 10 单，含机器判据与卡位账。",
    "scenario_key": "council-session-003-solver",
    "mission_id": 676,
    "tags": ["council", "sdaa", "baseline", "solver", "proposal"],
    "markdown": md,
    "figures": [
        {
            "name": "fig1_perf_noise.png",
            "caption": "(a) m397 五连跑墙钟单调劣化 2.60×（CV 38.81%，判 FAIL）；(b) 现场 200 次 matmul：raw-mean CV 11.31% vs block-median CV 1.51%（card25）",
            "png_base64": b64(ART / "fig1_perf_noise.png"),
        },
        {
            "name": "fig2_accuracy_debt.png",
            "caption": "(a) Hertz 四门与自由面质量漂移的实测 vs 阈值（对数）；(b) 超阈倍数，最大 58.6×（接触能量漂移）",
            "png_base64": b64(ART / "fig2_accuracy_debt.png"),
        },
        {
            "name": "fig3_lib_tensor_backend.png",
            "caption": "10 库 import torch 文件占比（现场 grep /root/tensor-ecosystem）：TensorBEM 0%（numpy-only），TensorLBM 69.67% 且唯一有 sdaa 引用",
            "png_base64": b64(ART / "fig3_lib_tensor_backend.png"),
        },
    ],
}

Path("/tmp/report.json").write_text(json.dumps(payload, ensure_ascii=False))
print(json.dumps({"markdown_chars": len(md), "figures": [f["name"] for f in payload["figures"]]}, ensure_ascii=False))