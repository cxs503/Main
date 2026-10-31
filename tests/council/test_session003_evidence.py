"""session-003 求解器槽位提案的证据自校验（提案 doc 内所有数字必须能从这里复算）。

跑法：pytest -q tests/council/test_session003_evidence.py
若源工件（m397/m621/m404/m407）不在本机，相关断言 skip，不失败（诚实标注）。
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
EV = ROOT / "artifacts" / "council_session003" / "evidence.json"
PROBE = ROOT / "artifacts" / "council_session003" / "live_matmul_probe_card25.json"

SRC = {
    "perf": "/root/tp-agent/work/m397/artifacts/sdaa_perf_baseline.json",
    "cc3": "/root/tp-agent/work/m621/artifacts/cross_card_3card.json",
    "cc4": "/root/tp-agent/work/m621/artifacts/cross_card_4card.json",
    "bench": "/root/tp-agent/work/m621/artifacts/bench.json",
    "hertz": "/root/tp-agent/work/m404/artifacts/dem_fem_hertz_cpu.json",
    "dam": "/root/tp-agent/work/m407/artifacts/sdaa_dambreak3d.json",
}


def load(p):
    with open(p) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def ev():
    assert EV.exists(), f"evidence.json 缺失: {EV}"
    return load(EV)


def need(key):
    p = SRC[key]
    if not Path(p).exists():
        pytest.skip(f"源工件不在本机: {p}")
    return load(p)


def cv(v):
    return statistics.pstdev(v) / statistics.mean(v) * 100.0


def test_live_probe_statistics(ev):
    """现场探测：raw-mean CV 与 block-median CV 必须都被如实记录，且后者显著更小。"""
    probe = load(PROBE)
    series = probe["series_ms"]
    assert len(series) == ev["live_probe"]["n"] == 200
    assert round(cv(series), 2) == pytest.approx(ev["live_probe"]["raw_mean_cv_pct"], abs=0.01)
    blocks = [statistics.median(series[i * 50:(i + 1) * 50]) for i in range(4)]
    assert [round(b, 3) for b in blocks] == ev["live_probe"]["block_medians_ms"]
    assert round(cv(blocks), 2) == pytest.approx(ev["live_probe"]["block_median_cv_pct"], abs=0.01)
    # 提案的核心论点：同一数据不同统计量判决相反
    assert ev["live_probe"]["raw_mean_cv_pct"] > 5.0
    assert ev["live_probe"]["block_median_cv_pct"] < 2.0


def test_perf_baseline_is_invalid_and_monotone(ev):
    """m397 基线：CV>2% 故无效，且 5 次连跑单调递增（不是白噪声）。"""
    perf = need("perf")
    wall = [r["wall_s"] for r in perf["runs"]]
    assert wall == ev["perf"]["wall_s"]
    # 工件用的是样本标准差 (n-1)；总体标准差口径另记在 cv_pct_population
    assert round(statistics.stdev(wall) / statistics.mean(wall) * 100, 2) == pytest.approx(ev["perf"]["raw_mean_cv_pct"], abs=0.01)
    assert round(cv(wall), 2) == pytest.approx(ev["perf"]["cv_pct_population"], abs=0.01)
    assert ev["perf"]["raw_mean_cv_pct"] > 2.0
    assert all(b > a for a, b in zip(wall, wall[1:]))
    assert ev["perf"]["monotone_increasing"] is True


def test_cross_card_determinism_and_perf_drift(ev):
    """多卡正确性 bitwise 一致，但单卡步时随可见卡数漂移（性能口径不稳的证据）。"""
    cc3, cc4 = need("cc3"), need("cc4")
    assert cc3["single_card"]["rho_sha256"] == cc4["single_card"]["rho_sha256"]
    assert cc3["single_card"]["ux_sha256"] == cc4["single_card"]["ux_sha256"]
    delta = (cc4["single_card"]["mean_step_ms"] / cc3["single_card"]["mean_step_ms"] - 1) * 100
    assert round(delta, 2) == pytest.approx(ev["cross_card"]["delta_pct"], abs=0.01)


def test_accuracy_debt_numbers(ev):
    """精度债表（Hertz 四门 + dambreak 漂移）逐项可复算，且全部未过门。"""
    h = need("hertz")["hertz"]
    d = need("dam")
    debt = ev["accuracy_debt"]
    assert h["fitting"]["fitting_exponent"] == pytest.approx(debt["hertz_exponent"])
    assert h["stiffness"]["stiffness_error_pct"] == pytest.approx(debt["hertz_stiffness_err_pct"])
    assert h["energy"]["energy_drift_pct"] == pytest.approx(debt["hertz_energy_drift_pct"])
    assert h["mesh_convergence"]["stiffness_change_pct"] == pytest.approx(debt["hertz_mesh_change_pct"])
    assert abs(d["cpu_baseline"]["drift_per_1e4_steps_pct"]) == pytest.approx(debt["dambreak_drift_per_1e4_pct"])
    # 全部超阈：门是"可能失败"的
    assert debt["hertz_stiffness_err_pct"] > debt["hertz_stiffness_target_pct"]
    assert debt["hertz_energy_drift_pct"] > debt["hertz_energy_target_pct"]
    assert debt["dambreak_drift_per_1e4_pct"] > debt["dambreak_target_per_1e4_pct"]
    assert abs(debt["hertz_exponent"] - debt["hertz_exponent_target"]) > debt["hertz_exponent_tol"]


def test_lib_matrix_covers_ten_libs(ev):
    """10 库矩阵齐全；TensorBEM 判 numpy-only；TensorLBM 是唯一有 sdaa 引用的库。"""
    rows = {r["lib"]: r for r in ev["lib_matrix"]}
    assert len(rows) >= 10
    assert rows["TensorBEM"]["torch_files"] == 0
    assert rows["TensorBEM"]["compute_backend"] == "numpy-only"
    sdaa_libs = [k for k, r in rows.items() if r["sdaa_files"] > 0]
    assert "TensorLBM" in sdaa_libs
    # 诚实边界：只有 LBM 有设备级痕迹
    assert [k for k in sdaa_libs if k != "TensorPlatform"] == ["TensorLBM"]


def test_proposal_doc_exists_and_is_consistent(ev):
    """提案正文必须存在、带 author-model 头，且引用的关键阈值一致。"""
    doc = ROOT / "docs" / "council" / "session-003" / "r1-proposal-solver.md"
    text = doc.read_text()
    assert text.splitlines()[0].startswith("author-model:")
    for token in ["38.81", "11.31", "1.51", "44.09", "4.476", "1.4108"]:
        assert token in text, f"提案缺少关键数字 {token}"
    # 提案中的 CV 门槛与 evidence 口径一致
    assert ev["live_probe"]["block_median_cv_pct"] <= 2.0