"""计数稳定性核查单元测试。

覆盖关键情形：
- 计数唯一（stable，含多解但计数相同）；
- 计数不唯一且存在可一判定音的未判断框对（ambiguous）；
- 计数不唯一但不存在合格框对（undetermined / no_pair）；
- 判断冲突不可行（undetermined / infeasible）；
- 框数超限（unsupported）；
- 超时（undetermined / timeout）；
- 穷举结论与 ILP 求解器的最优代价/计数互相印证。
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.example import build_example_detections
from app.solver import solve_tracking
from app.stability import MAX_CHECK_BOXES, check_count_stability

W_DIST, W_APP, PENALTY = 1.0, 10000.0, 15000.0


def mk(bid, frame, cx, cy, vec, size=10):
    return {
        "id": bid,
        "frame": frame,
        "rect": [cx - size / 2, cy - size / 2, size, size],
        "vector": list(vec),
    }


# ---------------------------------------------------------------- stable

def test_empty_is_stable_zero():
    r = check_count_stability([], [], W_DIST, W_APP, PENALTY)
    assert r["status"] == "stable"
    assert r["animal_count"] == 0


def test_chain_unique_count_stable():
    dets = [mk("a", 0, 10, 10, [1, 0]), mk("b", 1, 20, 10, [1, 0]),
            mk("c", 2, 30, 10, [1, 0])]
    r = check_count_stability(dets, [], W_DIST, W_APP, PENALTY)
    assert r["status"] == "stable"
    assert r["animal_count"] == 1
    assert r["n_optimal_solutions"] >= 1


def test_multiple_optima_same_count_is_stable():
    # 4 框：帧 0 的 a，帧 1 的 b、c（同帧，必属不同轨迹），帧 2 的 d。
    # b、c 关于 a-d 连线对称，故两种覆盖等代价且都含 2 条轨迹：
    #   {a→b→d, c 单独} 与 {a→c→d, b 单独}
    # 高罚分下 2 条轨迹最优 → 多解但计数恒为 2（stable）。
    dets = [
        mk("a", 0, 10, 0, [1, 0]),    # 中心 (10, 0)
        mk("b", 1, 20, 1, [1, 0]),    # 中心 (20, 1)
        mk("c", 1, 20, -1, [1, 0]),   # 中心 (20, -1)
        mk("d", 2, 30, 0, [1, 0]),    # 中心 (30, 0)
    ]
    r = check_count_stability(dets, [], 1.0, 0.0, 1000.0)
    assert r["status"] == "stable"
    assert r["animal_count"] == 2
    assert r["n_optimal_solutions"] >= 2  # 确实存在多条同代价最优轨迹


def test_example_stable_two_animals():
    dets = build_example_detections()  # 11 框 → 超限，先确认 unsupported
    r = check_count_stability(dets, [], W_DIST, W_APP, PENALTY)
    assert r["status"] == "unsupported"

    # 取前 6 个框（示例的前两帧 + 第 2 帧）构造 ≤6 框的可行子问题
    sub = [d for d in dets if d["frame"] <= 2]
    assert len(sub) == 6
    r2 = check_count_stability(sub, [], W_DIST, W_APP, PENALTY)
    assert r2["status"] == "stable"
    # 与 ILP 求解器在同一子问题上的结论一致
    sol = solve_tracking(sub, [], W_DIST, W_APP, PENALTY)
    assert sol["feasible"]
    assert r2["animal_count"] == sol["animal_count"]
    assert r2["optimal_cost"] == pytest.approx(sol["total_cost"], abs=1e-6)


# ---------------------------------------------------------------- ambiguous

def test_ambiguous_pair_distinguishes_two_counts():
    # 两框相邻帧：连接则 1 只，不连则 2 只；令两种方案总代价恰好相等
    dets = [mk("a", 0, 0, 0, [1, 0]), mk("b", 1, 10, 0, [0, 1])]
    # 边代价 = 1 * 100 + 10000 * (1 - 0) = 10100；罚分取相同值 → 两种方案等价
    r = check_count_stability(dets, [], 1.0, 10000.0, 10100.0)
    assert r["status"] == "ambiguous"
    pair = r["pair"]
    assert {pair["a"]["id"], pair["b"]["id"]} == {"a", "b"}
    assert {pair["same_count"], pair["different_count"]} == {1, 2}
    assert pair["same_count"] != pair["different_count"]
    assert r["distinct_counts"] == [1, 2]
    assert r["n_optimal_solutions"] == 2


def test_ambiguous_pair_must_be_unjudged():
    dets = [mk("a", 0, 0, 0, [1, 0]), mk("b", 1, 10, 0, [0, 1])]
    # 一旦该框对已被判断，核查不能再把它作为补判框对
    r_judged = check_count_stability(
        dets, [{"a": 0, "b": 1, "type": "same"}], 1.0, 10000.0, 10100.0)
    assert r_judged["status"] == "stable"
    assert r_judged["animal_count"] == 1


# ---------------------------------------------------------------- undetermined

def test_no_qualifying_pair_undetermined():
    # 3 框等边三角形（直线布局），罚分等于边代价：
    # 1 条/2 条/3 条轨迹的最小总代价全部相等（均为 2P）
    dets = [
        mk("a", 0, 0, 0, [1, 0]),
        mk("b", 1, 10, 0, [1, 0]),
        mk("c", 2, 20, 0, [1, 0]),
    ]
    p = 100.0  # 边代价 = 距离平方 100（外观相同）
    r = check_count_stability(dets, [], 1.0, 0.0, p)
    assert r["status"] == "undetermined"
    assert r["reason"] == "no_pair"
    assert r["distinct_counts"] == [1, 2, 3]


def test_infeasible_judgments_undetermined():
    dets = [mk("a", 0, 0, 0, [1, 0]), mk("b", 0, 50, 0, [1, 0])]
    r = check_count_stability(
        dets, [{"a": 0, "b": 1, "type": "same"}], W_DIST, W_APP, PENALTY)
    assert r["status"] == "undetermined"
    assert r["reason"] == "infeasible"
    assert "animal_count" not in r


def test_timeout_returns_undetermined():
    # 6 框、帧差布局使候选边较多；时间预算给 0，必在首批检查点超时
    dets = [mk(f"b{i}", i % 3, 10 * i, 10 * i, [1, 0]) for i in range(6)]
    r = check_count_stability(dets, [], W_DIST, W_APP, PENALTY, time_limit=0.0)
    assert r["status"] == "undetermined"
    assert r["reason"] == "timeout"
    assert "animal_count" not in r


# ---------------------------------------------------------------- 与 ILP 对照

def test_enumeration_matches_ilp_optimum():
    import random
    rng = random.Random(42)
    for _ in range(20):
        n = rng.randint(1, MAX_CHECK_BOXES)
        n_frames = rng.randint(2, 4)
        dets = [
            mk(f"b{k}", rng.randrange(n_frames), rng.uniform(0, 100),
               rng.uniform(0, 100), [rng.uniform(0.1, 1) for _ in range(3)])
            for k in range(n)
        ]
        judgments = []
        if n >= 2:
            for _ in range(rng.randint(0, 2)):
                a, b = rng.sample(range(n), 2)
                judgments.append({"a": a, "b": b, "type": rng.choice(["same", "different"])})
        w_dist, w_app, penalty = rng.uniform(0, 2), rng.uniform(0, 50), rng.uniform(0, 500)

        sol = solve_tracking(dets, judgments, w_dist, w_app, penalty)
        chk = check_count_stability(dets, judgments, w_dist, w_app, penalty)
        if not sol["feasible"]:
            assert chk["status"] == "undetermined" and chk["reason"] == "infeasible"
            continue
        assert chk["status"] in {"stable", "ambiguous", "undetermined"}
        assert chk["optimal_cost"] == pytest.approx(sol["total_cost"], rel=1e-7)
        # ILP 给出的计数必在穷举得到的可能计数集合内
        assert sol["animal_count"] in chk["distinct_counts"]
        if chk["status"] == "stable":
            assert chk["animal_count"] == sol["animal_count"]
