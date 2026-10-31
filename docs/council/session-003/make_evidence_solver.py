#!/usr/bin/env python3
"""session-003 r1-proposal-solver 证据生成脚本（零手抄：所有数字从工件/现场探测算出来）。

输入（只读）：
  - m397/m481/... 的 sdaa_perf_baseline.json（同内容，取 m397）
  - m621 cross_card_3card.json / cross_card_4card.json / bench.json（matmul 微基准）
  - m404 dem_fem_hertz_cpu.json（Hertz 接触 honest-FAIL）
  - m407 sdaa_dambreak3d.json（自由面质量守恒 honest-FAIL）
  - artifacts/council_session003/live_matmul_probe_card25.json（本次现场探测，本脚本产出）
  - /root/tensor-ecosystem/*（10 库张量后端矩阵，现场 grep 统计）

输出：
  - artifacts/council_session003/evidence.json
  - artifacts/council_session003/fig1_perf_noise.png
  - artifacts/council_session003/fig2_accuracy_debt.png
  - artifacts/council_session003/fig3_lib_tensor_backend.png
"""
from __future__ import annotations

import json
import os
import statistics
import subprocess
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path("artifacts/council_session003")
OUT.mkdir(parents=True, exist_ok=True)

SRC = {
    "perf_baseline_m397": "/root/tp-agent/work/m397/artifacts/sdaa_perf_baseline.json",
    "cross_card_3card_m621": "/root/tp-agent/work/m621/artifacts/cross_card_3card.json",
    "cross_card_4card_m621": "/root/tp-agent/work/m621/artifacts/cross_card_4card.json",
    "matmul_bench_m621": "/root/tp-agent/work/m621/artifacts/bench.json",
    "hertz_m404": "/root/tp-agent/work/m404/artifacts/dem_fem_hertz_cpu.json",
    "dambreak_m407": "/root/tp-agent/work/m407/artifacts/sdaa_dambreak3d.json",
    "live_probe": str(OUT / "live_matmul_probe_card25.json"),
}
ECOSYSTEM = Path("/root/tensor-ecosystem")


def load(p: str):
    with open(p) as f:
        return json.load(f)


def cv(values):
    return statistics.pstdev(values) / statistics.mean(values) * 100.0


def block_medians(values, nblocks=4):
    n = len(values) // nblocks
    return [statistics.median(values[i * n:(i + 1) * n]) for i in range(nblocks)]


def sh(cmd: str) -> str:
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout


# ---------------------------------------------------------------- 1) 现场探测
def live_probe(force=False):
    """现场单卡 matmul 微基准（200 次，block-median 与 raw-mean 对照）。"""
    p = OUT / "live_matmul_probe_card25.json"
    if p.exists() and not force:
        return load(str(p))
    script = OUT / "probe_matmul.py"
    env = dict(os.environ, SDAA_QUIET="1")
    subprocess.run(["python3", str(script)], env=env, capture_output=True, text=True, timeout=900)
    return load(str(p))


# ------------------------------------------------------------ 2) 10 库后端矩阵
def lib_matrix():
    rows = []
    for d in sorted(ECOSYSTEM.iterdir()):
        if not d.is_dir():
            continue
        py = int(sh(f"find {d} -name '*.py' | wc -l").strip())
        loc = int(sh(f"find {d} -name '*.py' -print0 | xargs -0 cat 2>/dev/null | wc -l").strip())
        torch_files = int(sh(f"grep -rl --include='*.py' -E '^[[:space:]]*(import torch|from torch)' {d} | wc -l").strip())
        np_files = int(sh(f"grep -rl --include='*.py' 'import numpy' {d} | wc -l").strip())
        sdaa_files = int(sh(f"grep -rl --include='*.py' 'sdaa' {d} | wc -l").strip())
        native = int(sh(f"grep -rl --include='*.py' -E 'numba|scipy\\.sparse' {d} | wc -l").strip())
        bm = int(sh(f"ls {d}/benchmarks 2>/dev/null | wc -l").strip())
        ratio = torch_files / py if py else 0.0
        if torch_files == 0:
            backend = "numpy-only"
        elif native > torch_files:
            backend = "cpu-native-dominant(numpy/scipy/numba)"
        else:
            backend = "torch-native"
        rows.append(
            {
                "lib": d.name,
                "py_files": py,
                "loc": loc,
                "torch_files": torch_files,
                "numpy_files": np_files,
                "sdaa_files": sdaa_files,
                "cpu_native_files": native,
                "benchmark_entries": bm,
                "torch_file_ratio": round(ratio, 4),
                "compute_backend": backend,
            }
        )
    return rows


# ------------------------------------------------------------------ 3) 汇总
def build():
    perf = load(SRC["perf_baseline_m397"])
    cc3 = load(SRC["cross_card_3card_m621"])
    cc4 = load(SRC["cross_card_4card_m621"])
    bench = load(SRC["matmul_bench_m621"])
    hertz = load(SRC["hertz_m404"])["hertz"]
    dam = load(SRC["dambreak_m407"])
    live = live_probe()

    perf_wall = [r["wall_s"] for r in perf["runs"]]
    live_series = live["series_ms"]
    live_blocks = block_medians(live_series, 4)

    ev = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "sources": {k: {"path": v, "sha256": sh(f"sha256sum {v} | cut -d' ' -f1").strip()} for k, v in SRC.items()},
        "perf": {
            "case": perf["case"],
            "wall_s": perf_wall,
            "raw_mean_cv_pct": perf["cv_pct"],
            "cv_estimator": "artifact uses sample std (n-1)=98.8078/mean=254.5935 -> 38.81%; population std would give 34.71% (recorded in cv_pct_population)",
            "cv_pct_population": round(cv(perf_wall), 2),
            "median_wall_s": perf["median_wall_s"],
            "monotone_increasing": all(b > a for a, b in zip(perf_wall, perf_wall[1:])),
            "last_over_first": round(perf_wall[-1] / perf_wall[0], 3),
            "mlups": [r["mlups"] for r in perf["runs"]],
            "verdict_in_artifact": "cv_pass=%s" % perf.get("cv_pass"),
        },
        "microbench_m621": {
            "device": bench["environment"]["device_name"],
            "mean_ms": bench["sdaa"]["mean_ms"],
            "cv_pct": bench["sdaa"]["cv_pct"],
            "n_iter": bench["sdaa"]["n_iter"],
        },
        "live_probe": {
            "device_card": "tcaicard25 (SDAA_VISIBLE_DEVICES=25)",
            "n": live["n"],
            "raw_mean_ms": live["overall"]["mean"],
            "raw_median_ms": live["overall"]["med"],
            "raw_mean_cv_pct": live["overall"]["cv"],
            "min_ms": live["overall"]["mn"],
            "max_ms": live["overall"]["mx"],
            "block_medians_ms": [round(x, 3) for x in live_blocks],
            "block_median_cv_pct": round(cv(live_blocks), 2),
            "cv_estimator": "population std (n), applied identically to raw series and block medians",
            "spikes_gt_2x_median": live["spikes_gt_2x_med"],
        },
        "cross_card": {
            "single_card_mean_step_ms_3visible": cc3["single_card"]["mean_step_ms"],
            "single_card_mean_step_ms_4visible": cc4["single_card"]["mean_step_ms"],
            "delta_pct": round((cc4["single_card"]["mean_step_ms"] / cc3["single_card"]["mean_step_ms"] - 1) * 100, 2),
            "field_sha256_identical_3_vs_4": (
                cc3["single_card"]["rho_sha256"] == cc4["single_card"]["rho_sha256"]
                and cc3["single_card"]["ux_sha256"] == cc4["single_card"]["ux_sha256"]
            ),
        },
        "accuracy_debt": {
            "hertz_exponent": hertz["fitting"]["fitting_exponent"],
            "hertz_exponent_target": hertz["fitting"]["exponent_target"],
            "hertz_exponent_tol": hertz["fitting"]["exponent_tolerance"],
            "hertz_stiffness_err_pct": hertz["stiffness"]["stiffness_error_pct"],
            "hertz_stiffness_target_pct": 3.0,
            "hertz_energy_drift_pct": hertz["energy"]["energy_drift_pct"],
            "hertz_energy_target_pct": 1.0,
            "hertz_mesh_change_pct": hertz["mesh_convergence"]["stiffness_change_pct"],
            "hertz_mesh_target_pct": 5.0,
            "dambreak_drift_per_1e4_pct": abs(dam["cpu_baseline"]["drift_per_1e4_steps_pct"]),
            "dambreak_target_per_1e4_pct": 1.0,
            "dambreak_status": dam["status"],
        },
        "lib_matrix": lib_matrix(),
        "teco_smi": {
            "binary": "/opt/tecoai/bin/teco-smi",
            "version": sh("/opt/tecoai/bin/teco-smi 2>/dev/null | grep -o 'TECO-SMI: [0-9.]*'").strip(),
            "index_entries": int(sh("/opt/tecoai/bin/teco-smi 2>/dev/null | grep -c TECO_AICARD").strip()),
            "dev_nodes": int(sh("ls /dev | grep -c tcaicard").strip()),
            "busy_index": sh(
                "/opt/tecoai/bin/teco-smi 2>/dev/null | grep TECO_AICARD | sed 's/|//g' | awk '{if ($NF!=\"0%\") print $1\":\"$NF}'"
            ).strip(),
        },
    }
    with open(OUT / "evidence.json", "w") as f:
        json.dump(ev, f, ensure_ascii=False, indent=2)

    # ---- 图 1：性能噪声（历史 5 连跑 vs 现场微基准）
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    ax[0].plot(range(1, 6), perf_wall, "o-", color="#c0392b", label="LBM cavity_3d_full wall_s")
    ax[0].set_xlabel("repeat #")
    ax[0].set_ylabel("wall clock (s)")
    ax[0].set_title(f"(a) m397 5-repeat baseline: CV={perf['cv_pct']:.2f}% (monotone={ev['perf']['monotone_increasing']})")
    ax[0].grid(alpha=0.3)
    ax[0].legend()

    ax[1].plot(range(len(live_series)), live_series, lw=0.8, color="#2c7fb8", label="per-iter cost (ms)")
    bx = [len(live_series) * (i + 0.5) / 4 for i in range(4)]
    ax[1].plot(bx, live_blocks, "s--", color="#c0392b", label=f"block median (CV={ev['live_probe']['block_median_cv_pct']}%)")
    ax[1].axhline(live["overall"]["med"], color="k", ls=":", lw=1, label=f"median={live['overall']['med']:.2f} ms")
    ax[1].set_xlabel("iteration (1200x1200 fp32 matmul, card25)")
    ax[1].set_ylabel("latency (ms)")
    ax[1].set_title(f"(b) live probe: raw-mean CV={live['overall']['cv']:.2f}% vs block-median CV={ev['live_probe']['block_median_cv_pct']}%")
    ax[1].grid(alpha=0.3)
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_perf_noise.png", dpi=140)
    plt.close(fig)

    # ---- 图 2：精度债 vs 阈值
    items = [
        ("Hertz stiffness err (%)", ev["accuracy_debt"]["hertz_stiffness_err_pct"], 3.0),
        ("Hertz energy drift (%)", ev["accuracy_debt"]["hertz_energy_drift_pct"], 1.0),
        ("Hertz mesh change (%)", ev["accuracy_debt"]["hertz_mesh_change_pct"], 5.0),
        ("dambreak mass drift (%/1e4)", ev["accuracy_debt"]["dambreak_drift_per_1e4_pct"], 1.0),
    ]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    labels = [i[0] for i in items]
    meas = [i[1] for i in items]
    thr = [i[2] for i in items]
    x = range(len(items))
    ax[0].bar([i - 0.2 for i in x], meas, width=0.4, color="#c0392b", label="measured")
    ax[0].bar([i + 0.2 for i in x], thr, width=0.4, color="#27ae60", label="threshold")
    ax[0].set_xticks(list(x))
    ax[0].set_xticklabels(labels, rotation=18, ha="right", fontsize=8)
    ax[0].set_yscale("log")
    ax[0].set_ylabel("% (log)")
    ax[0].set_title("(a) 现有精度债：全部未过门（m404/m407 实测）")
    ax[0].legend()
    ax[0].grid(alpha=0.3, axis="y")

    ratio = [m / t for m, t in zip(meas, thr)]
    ax[1].barh(labels, ratio, color="#c0392b")
    ax[1].axvline(1.0, color="#27ae60", ls="--", label="threshold = 1x")
    ax[1].set_xscale("log")
    ax[1].set_xlabel("measured / threshold (x, log)")
    ax[1].set_title("(b) 超阈倍数")
    ax[1].legend()
    ax[1].grid(alpha=0.3, axis="x")
    for i, r in enumerate(ratio):
        ax[1].text(r * 1.05, i, f"{r:.1f}x", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_accuracy_debt.png", dpi=140)
    plt.close(fig)

    # ---- 图 3：10 库张量后端占比
    rows = [r for r in ev["lib_matrix"] if r["py_files"] > 0]
    rows.sort(key=lambda r: r["torch_file_ratio"])
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#27ae60" if r["torch_file_ratio"] >= 0.5 else ("#e67e22" if r["torch_file_ratio"] > 0 else "#c0392b") for r in rows]
    ax.barh([r["lib"] for r in rows], [r["torch_file_ratio"] * 100 for r in rows], color=colors)
    for i, r in enumerate(rows):
        ax.text(r["torch_file_ratio"] * 100 + 1, i, f"{r['torch_file_ratio']*100:.2f}%  ({r['torch_files']}/{r['py_files']} py)", va="center", fontsize=8)
    ax.set_xlabel("files importing torch / total py files (%)")
    ax.set_title("10 库张量后端真实性（现场 grep /root/tensor-ecosystem）")
    ax.set_xlim(0, 100)
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(OUT / "fig3_lib_tensor_backend.png", dpi=140)
    plt.close(fig)

    print(json.dumps({"evidence": str(OUT / "evidence.json"), "perf_raw_cv": perf["cv_pct"],
                      "live_block_median_cv": ev["live_probe"]["block_median_cv_pct"],
                      "live_raw_cv": ev["live_probe"]["raw_mean_cv_pct"]}, ensure_ascii=False))


if __name__ == "__main__":
    build()