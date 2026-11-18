"""计数稳定性核查：枚举满足身份与跨帧约束的全部全局最优轨迹。

背景
----
同一份相机陷阱影像在同一最小代价下可能存在多条全局最优轨迹，而不同最优
轨迹的轨迹条数（= 动物计数）可能不同。本模块对**至多 6 个检测框**的会话
穷举所有合法路径覆盖，给出计数是否唯一确定的结论。

方法
----
n ≤ 6 时候选边很少，直接对“每个框选择一个后继（或成为轨迹起点）”做深度
优先枚举：边仅由较早帧指向较晚帧（帧差 1~2），故选中边天然无环；枚举中
再限制每个框入度 ≤ 1，即得到全部点不相交路径覆盖。在叶节点复用求解器的
同只/异只约束语义（单位流可达性与割约束等价于“同只须同轨、异只须异轨”）
过滤，收集代价达到全局最小的所有覆盖，并：

- 若所有最优覆盖的轨迹条数相同 → 计数唯一（stable）；
- 否则从**尚未人工判断**的框对中找一对，使“同只”和“异只”在原最优解集合中
  都可能成立，且两种回答各自只对应一个不同的动物数（ambiguous），
  供人工补判；
- 找不到这样的框对、核查超时或输入不可行 → 计数仍未确定（undetermined）。
"""

from __future__ import annotations

import time

from .solver import InputConflict, _build_edges, _same_classes

# 只对至多 6 个框的会话做穷举核查
MAX_CHECK_BOXES = 6
# 穷举核查的默认时间预算（秒）
DEFAULT_TIME_LIMIT = 30.0
# 代价相等的浮点容差
_COST_TOL = 1e-9


class CheckTimeout(Exception):
    """穷举核查超出时间预算。"""


def check_count_stability(detections, judgments, w_dist, w_app, penalty,
                          time_limit=DEFAULT_TIME_LIMIT):
    """核查全局最优解的动物计数是否唯一确定。

    参数与 :func:`app.solver.solve_tracking` 一致（judgments 使用框下标）。
    返回结构化字典，``status`` 取值：

    - ``stable``      ：所有最优轨迹计数一致，给出唯一最优计数；
    - ``ambiguous``   ：计数不唯一，给出可补判的框对与“同只/异只”两个计数；
    - ``undetermined``：无合格框对、超时或输入不可行，计数仍未确定；
    - ``unsupported`` ：框数超过 :data:`MAX_CHECK_BOXES`，不做穷举。
    """
    n = len(detections)
    if n == 0:
        return {
            "status": "stable",
            "message": "无检测框，唯一最优计数为 0。",
            "animal_count": 0,
            "n_optimal_solutions": 1,
            "distinct_counts": [0],
        }
    if n > MAX_CHECK_BOXES:
        return {
            "status": "unsupported",
            "message": (
                f"计数稳定性核查仅支持至多 {MAX_CHECK_BOXES} 个检测框的会话，"
                f"当前 {n} 个框，未做穷举核查。"
            ),
        }

    # 枚举假设候选边一律由小下标指向大下标（框按帧升序）。
    # 入口处统一规范化：重排检测框并同步重映射判断下标。
    order = sorted(range(n), key=lambda i: (int(detections[i]["frame"]),
                                            str(detections[i]["id"])))
    if order != list(range(n)):
        remap = {old: new for new, old in enumerate(order)}
        detections = [detections[old] for old in order]
        judgments = [
            {"a": remap[j["a"]], "b": remap[j["b"]], "type": j["type"]}
            for j in judgments
        ]

    try:
        edges, frames = _build_edges(detections, w_dist, w_app)
        classes, diff_pairs = _same_classes(n, judgments, frames,
                                            [str(d["id"]) for d in detections])
    except InputConflict as exc:
        return {
            "status": "undetermined",
            "reason": "infeasible",
            "message": f"当前判断不可行（{exc}），不存在可行轨迹，计数未确定。",
        }

    deadline = time.monotonic() + float(time_limit)
    deadline_ticks = 0

    out_edges = [[] for _ in range(n)]
    for k, (i, j, _) in enumerate(edges):
        out_edges[i].append(k)

    # 已判断框对（下标对，规范化为小下标在前）
    judged_pairs = {tuple(sorted((j["a"], j["b"]))) for j in judgments}

    succ = [-1] * n          # 每框选中的后继下标，-1 表示无后继
    taken: set[int] = set()  # 已被选为后继的框（入度 = 1）
    optimal = []             # [(cost, n_traj, succ_tuple)]
    best_cost = None

    def cost_tol(cost):
        return max(_COST_TOL, _COST_TOL * max(1.0, abs(cost)))

    def reachable(a, b, succ_map):
        cur = a
        while cur in succ_map:
            cur = succ_map[cur]
            if cur == b:
                return True
        return False

    def leaf(total_cost):
        """叶节点：校验身份/异只约束并维护最优解集合。"""
        nonlocal best_cost
        succ_map = {i: j for i, j in enumerate(succ) if j >= 0}
        # 同只：类内最早框必须能到达其余成员（即同一条轨迹）
        for members in classes:
            root = members[0]
            for m in members[1:]:
                if not reachable(root, m, succ_map):
                    return
        # 异只：两框不得在同一条轨迹上
        for a, b in diff_pairs:
            if reachable(a, b, succ_map):
                return
        n_traj = n - len(succ_map)
        if best_cost is None or total_cost < best_cost - cost_tol(total_cost):
            optimal.clear()
            optimal.append((total_cost, n_traj, tuple(succ)))
            best_cost = total_cost
        elif abs(total_cost - best_cost) <= cost_tol(best_cost):
            optimal.append((total_cost, n_traj, tuple(succ)))

    def enumerate_covers(i, assoc_cost):
        """按框下标（帧升序）逐个决定后继。

        ``taken`` 记录已被某个更早框选作后继的框（每框至多一个前驱，
        因候选边一律由小下标指向大下标，处理 i 时其前驱只可能来自 < i）。
        回溯时仅移除本层置位的标记，兄弟子树之间不会互相污染。
        """
        nonlocal best_cost, deadline_ticks
        deadline_ticks += 1
        if (deadline_ticks & 63) == 0 and time.monotonic() > deadline:
            raise CheckTimeout
        if i == n:
            n_traj = n - sum(1 for j in succ if j >= 0)
            leaf(assoc_cost + penalty * n_traj)
            return
        # 选项 1：i 不选后继（i 是否为轨迹起点取决于它是否在 taken 中）
        succ[i] = -1
        enumerate_covers(i + 1, assoc_cost)
        # 选项 2：选择一条出边作为后继（目标框尚无入边）
        for k in out_edges[i]:
            _, j, edge_cost = edges[k]
            if j in taken:
                continue
            succ[i] = j
            taken.add(j)
            enumerate_covers(i + 1, assoc_cost + edge_cost)
            taken.discard(j)
        succ[i] = -1

    try:
        enumerate_covers(0, 0.0)
    except CheckTimeout:
        return {
            "status": "undetermined",
            "reason": "timeout",
            "message": f"计数稳定性核查在 {float(time_limit):.0f} 秒内未完成，计数仍未确定。",
        }

    if not optimal:
        # 输入层未发现冲突、但穷举证明无满足全部判断的路径覆盖
        return {
            "status": "undetermined",
            "reason": "infeasible",
            "message": "当前判断相互冲突，不存在满足全部约束的轨迹，计数未确定。",
        }

    distinct_counts = sorted({n_traj for _, n_traj, _ in optimal})
    n_solutions = len(optimal)
    optimal_cost = optimal[0][0]

    if len(distinct_counts) == 1:
        return {
            "status": "stable",
            "message": (
                f"已穷举全部 {n_solutions} 条代价相同的全局最优轨迹，"
                f"动物计数唯一确定。"
            ),
            "animal_count": distinct_counts[0],
            "n_optimal_solutions": n_solutions,
            "optimal_cost": float(optimal_cost),
            "distinct_counts": distinct_counts,
        }

    # 计数不唯一：在未判断框对中寻找可一判定音的补判框对
    def traj_id_of(succ_tuple):
        """由后继数组返回每框所属轨迹的代表点（轨迹起点下标）。

        边一律由较早帧指向较晚帧，框已按帧排序，故 i→j 必有 i<j，
        一次正向扫描即可把起点传播到整条轨迹。
        """
        rep = list(range(n))
        for i, j in enumerate(succ_tuple):
            if j >= 0:
                rep[j] = rep[i]
        return rep

    reps = [traj_id_of(st) for _, _, st in optimal]

    candidate = None
    for a in range(n):
        for b in range(a + 1, n):
            if (a, b) in judged_pairs:
                continue
            same_counts, diff_counts = set(), set()
            for rep, (_, n_traj, _) in zip(reps, optimal):
                if rep[a] == rep[b]:
                    same_counts.add(n_traj)
                else:
                    diff_counts.add(n_traj)
            # 两种回答在原最优解中都可能成立，且各自只对应一个不同计数
            if (len(same_counts) == 1 and len(diff_counts) == 1
                    and same_counts != diff_counts):
                candidate = (a, b, next(iter(same_counts)),
                             next(iter(diff_counts)))
                break
        if candidate is not None:
            break

    if candidate is None:
        return {
            "status": "undetermined",
            "reason": "no_pair",
            "message": (
                f"存在 {n_solutions} 条代价相同的全局最优轨迹，动物计数在 "
                f"{distinct_counts[0]}~{distinct_counts[-1]} 之间不唯一；"
                "但没有任何未判断框对能通过一次补判区分计数，计数仍未确定。"
            ),
            "n_optimal_solutions": n_solutions,
            "optimal_cost": float(optimal_cost),
            "distinct_counts": distinct_counts,
        }

    a, b, same_count, diff_count = candidate
    ids = [str(d["id"]) for d in detections]
    return {
        "status": "ambiguous",
        "message": (
            f"存在 {n_solutions} 条代价相同的全局最优轨迹，动物计数不唯一"
            f"（可能为 {('、'.join(str(c) for c in distinct_counts))}）；"
            "原计数标为暂定。请对下列框对补判，两种回答各自唯一确定一个计数。"
        ),
        "n_optimal_solutions": n_solutions,
        "optimal_cost": float(optimal_cost),
        "distinct_counts": distinct_counts,
        "pair": {
            "a": {"id": ids[a], "frame": frames[a]},
            "b": {"id": ids[b], "frame": frames[b]},
            "same_count": same_count,
            "different_count": diff_count,
        },
    }
