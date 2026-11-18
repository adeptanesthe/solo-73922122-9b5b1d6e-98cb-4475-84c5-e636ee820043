"""API 端到端测试：会话创建、校验、判断/参数版本、异步求解与迟到结果作废。"""

import io
import json
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.main import app

client = TestClient(app)


def png_bytes(color=(200, 30, 30)):
    img = Image.new("RGB", (64, 48), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def make_session(n_imgs=3, dets=None):
    if dets is None:
        dets = [
            {"id": "a", "frame": 0, "rect": [1, 1, 8, 8], "vector": [1, 0]},
            {"id": "b", "frame": 1, "rect": [11, 1, 8, 8], "vector": [1, 0]},
        ]
    files = [("images", (f"f{i}.png", png_bytes(), "image/png")) for i in range(n_imgs)]
    files.append(("detections", ("det.json", json.dumps({"detections": dets}), "application/json")))
    r = client.post("/api/sessions", files=files)
    assert r.status_code == 200, r.text
    return r.json()


def wait_job(job_id, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] != "running":
            return job
        time.sleep(0.05)
    raise TimeoutError("求解任务超时未完成")


# ---------------------------------------------------------------- 会话与校验

def test_create_session_ok():
    p = make_session()
    assert p["n_frames"] == 3 and len(p["detections"]) == 2
    assert p["version"] == 1
    img = client.get(f"/api/sessions/{p['session_id']}/frames/0")
    assert img.status_code == 200 and img.headers["content-type"] == "image/png"


def test_too_many_images_rejected():
    files = [("images", (f"f{i}.png", png_bytes(), "image/png")) for i in range(11)]
    files.append(("detections", ("d.json", json.dumps([]), "application/json")))
    r = client.post("/api/sessions", files=files)
    assert r.status_code == 400 and "10" in r.json()["detail"]


def test_too_many_detections_rejected():
    dets = [{"id": f"b{i}", "frame": 0, "rect": [0, 0, 4, 4], "vector": [1]} for i in range(31)]
    files = [("images", ("f0.png", png_bytes(), "image/png")),
             ("detections", ("d.json", json.dumps(dets), "application/json"))]
    r = client.post("/api/sessions", files=files)
    assert r.status_code == 400 and "30" in r.json()["detail"]


@pytest.mark.parametrize("det,needle", [
    ({"id": "a", "frame": 0, "rect": [0, 0, 4, 4], "vector": [0, 0]}, "非零"),
    ({"id": "a", "frame": 5, "rect": [0, 0, 4, 4], "vector": [1]}, "帧号"),
    ({"id": "a", "frame": 0, "rect": [0, 0, -4, 4], "vector": [1]}, "宽"),
    ({"id": "a", "frame": 0, "rect": [0, 0, 4, 4], "vector": [1, float("nan")]}, "非法"),
])
def test_bad_detection_rejected(det, needle):
    files = [("images", ("f0.png", png_bytes(), "image/png")),
             ("detections", ("d.json", json.dumps([det]), "application/json"))]
    r = client.post("/api/sessions", files=files)
    assert r.status_code == 400 and needle in r.json()["detail"]


def test_vector_dim_mismatch_rejected():
    dets = [
        {"id": "a", "frame": 0, "rect": [0, 0, 4, 4], "vector": [1, 0]},
        {"id": "b", "frame": 0, "rect": [8, 0, 4, 4], "vector": [1, 0, 0]},
    ]
    files = [("images", ("f0.png", png_bytes(), "image/png")),
             ("detections", ("d.json", json.dumps(dets), "application/json"))]
    r = client.post("/api/sessions", files=files)
    assert r.status_code == 400 and "维度" in r.json()["detail"]


# ---------------------------------------------------------------- 示例与求解

def test_example_solves_to_two_animals():
    p = client.post("/api/example").json()
    assert p["n_frames"] == 6 and len(p["detections"]) == 11
    job = client.post(f"/api/sessions/{p['session_id']}/solve").json()
    done = wait_job(job["job_id"])
    assert done["status"] == "done"
    res = done["result"]
    assert res["feasible"] and res["animal_count"] == 2 and res["n_trajectories"] == 2
    assert res["total_cost"] == pytest.approx(res["assoc_cost"] + res["penalty_cost"])


def test_conflicting_judgments_report_infeasible_without_count():
    p = client.post("/api/example").json()
    sid = p["session_id"]
    # A0 与 B0 同处第 0 帧，标记同只 → 无解
    r = client.put(f"/api/sessions/{sid}/judgments",
                   json={"judgments": [{"a": "A0", "b": "B0", "type": "same"}]})
    assert r.status_code == 200
    job = client.post(f"/api/sessions/{sid}/solve").json()
    done = wait_job(job["job_id"])
    res = done["result"]
    assert res["feasible"] is False
    assert "animal_count" not in res
    assert "无解" in res["message"]


def test_same_judgment_changes_result():
    p = client.post("/api/example").json()
    sid = p["session_id"]
    # 把 B0 与 A1 判为同只（它们帧差 1），全局解必须让二者同轨
    client.put(f"/api/sessions/{sid}/judgments",
               json={"judgments": [{"a": "B0", "b": "A1", "type": "same"}]})
    done = wait_job(client.post(f"/api/sessions/{sid}/solve").json()["job_id"])
    res = done["result"]
    assert res["feasible"]
    for t in res["trajectories"]:
        ids = {b["id"] for b in t["boxes"]}
        assert not ({"B0", "A1"} <= ids) or True  # 占位，下面统一检查
    same_traj = [t for t in res["trajectories"]
                 if {"B0", "A1"} <= {b["id"] for b in t["boxes"]}]
    assert len(same_traj) == 1


# ---------------------------------------------------------------- 版本与迟到结果

def test_version_bumps_and_stale_job_detectable():
    p = make_session()
    sid = p["session_id"]
    assert p["version"] == 1

    r = client.put(f"/api/sessions/{sid}/params",
                   json={"w_dist": 2.0, "w_app": 50.0, "penalty": 100.0})
    assert r.status_code == 200 and r.json()["version"] == 2

    job = client.post(f"/api/sessions/{sid}/solve").json()
    assert job["version"] == 2  # 任务记录启动时版本

    # 求解期间继续修改 → 版本再 +1，旧任务结果即为“迟到结果”
    r = client.put(f"/api/sessions/{sid}/judgments",
                   json={"judgments": [{"a": "a", "b": "b", "type": "different"}]})
    assert r.json()["version"] == 3

    done = wait_job(job["job_id"])
    assert done["version"] == 2  # 结果仍携带旧版本号
    current = client.get(f"/api/sessions/{sid}").json()
    assert current["version"] == 3
    assert done["version"] != current["version"]  # 前端据此丢弃，不覆盖新输入


def test_params_validation():
    p = make_session()
    r = client.put(f"/api/sessions/{p['session_id']}/params",
                   json={"w_dist": -1, "w_app": 1, "penalty": 1})
    assert r.status_code == 422


def test_unknown_session_404():
    assert client.get("/api/sessions/nope").status_code == 404
    assert client.post("/api/sessions/nope/solve").status_code == 404


def test_healthz():
    assert client.get("/healthz").json() == {"ok": True}


# ---------------------------------------------------------------- 计数稳定性核查

def _symmetric_dets():
    # 4 框对称场景：多条等代价最优轨迹，但计数恒为 2
    def box(bid, frame, cx, cy):
        return {"id": bid, "frame": frame, "rect": [cx - 5, cy - 5, 10, 10],
                "vector": [1, 0]}
    return [
        box("a", 0, 10, 0), box("b", 1, 20, 1),
        box("c", 1, 20, -1), box("d", 2, 30, 0),
    ]


def _ambiguous_dets():
    # 2 框：连接（1 只）与不连（2 只）总代价相等。
    # 中心分别为 (5,5)、(15,5)，距离平方 = 100；外观正交 → 外观项 10000。
    return [
        {"id": "a", "frame": 0, "rect": [0, 0, 10, 10], "vector": [1, 0]},
        {"id": "b", "frame": 1, "rect": [10, 0, 10, 10], "vector": [0, 1]},
    ]


def test_check_stability_stable_flow():
    p = make_session(n_imgs=3, dets=_symmetric_dets())
    sid = p["session_id"]
    done = wait_job(client.post(f"/api/sessions/{sid}/solve").json()["job_id"])
    assert done["result"]["feasible"]

    r = client.post(f"/api/sessions/{sid}/check-stability")
    assert r.status_code == 200
    job = wait_job(r.json()["job_id"])
    assert job["kind"] == "check" and job["status"] == "done"
    chk = job["result"]
    assert chk["status"] == "stable"
    assert chk["animal_count"] == done["result"]["animal_count"]
    assert chk["n_optimal_solutions"] >= 2


def test_check_stability_ambiguous_returns_pair_and_two_counts():
    p = make_session(n_imgs=2, dets=_ambiguous_dets())
    sid = p["session_id"]
    # 边代价 = 1*100 + 10000*(1-0) = 10100，罚分取同值 → 两种方案等价
    client.put(f"/api/sessions/{sid}/params",
               json={"w_dist": 1.0, "w_app": 10000.0, "penalty": 10100.0})
    done = wait_job(client.post(f"/api/sessions/{sid}/solve").json()["job_id"])
    assert done["result"]["feasible"]

    job = wait_job(client.post(f"/api/sessions/{sid}/check-stability").json()["job_id"])
    chk = job["result"]
    assert chk["status"] == "ambiguous"
    pair = chk["pair"]
    assert {pair["a"]["id"], pair["b"]["id"]} == {"a", "b"}
    assert {pair["same_count"], pair["different_count"]} == {1, 2}
    assert chk["distinct_counts"] == [1, 2]


def test_check_unsupported_when_more_than_six_boxes():
    dets = [
        {"id": f"b{i}", "frame": i % 3, "rect": [i, i, 4, 4], "vector": [1, 0]}
        for i in range(7)
    ]
    p = make_session(n_imgs=3, dets=dets)
    r = client.post(f"/api/sessions/{p['session_id']}/check-stability").json()
    assert r["job_id"] is None
    assert r["result"]["status"] == "unsupported"


def test_check_infeasible_judgments_undetermined():
    # 3 框链式布局，same+same+different 传递冲突 → 不可行
    dets = [
        {"id": "a", "frame": 0, "rect": [0, 0, 10, 10], "vector": [1, 0]},
        {"id": "b", "frame": 1, "rect": [10, 0, 10, 10], "vector": [1, 0]},
        {"id": "c", "frame": 2, "rect": [20, 0, 10, 10], "vector": [1, 0]},
    ]
    p = make_session(n_imgs=3, dets=dets)
    sid = p["session_id"]
    client.put(f"/api/sessions/{sid}/judgments", json={"judgments": [
        {"a": "a", "b": "b", "type": "same"},
        {"a": "b", "b": "c", "type": "same"},
        {"a": "a", "b": "c", "type": "different"},
    ]})
    solve_job = wait_job(client.post(f"/api/sessions/{sid}/solve").json()["job_id"])
    assert solve_job["result"]["feasible"] is False  # 原求解仍报告无解、不计数

    job = wait_job(client.post(f"/api/sessions/{sid}/check-stability").json()["job_id"])
    chk = job["result"]
    assert chk["status"] == "undetermined"
    assert chk["reason"] == "infeasible"
    assert "animal_count" not in chk


def test_check_result_invalidated_by_judgment_change():
    p = make_session(n_imgs=3, dets=_symmetric_dets())
    sid = p["session_id"]
    job0 = client.post(f"/api/sessions/{sid}/check-stability").json()
    # 核查运行期间改判 → 版本号前进
    client.put(f"/api/sessions/{sid}/judgments",
               json={"judgments": [{"a": "a", "b": "b", "type": "different"}]})
    done = wait_job(job0["job_id"])
    # 迟到结果仍带旧版本号，前端可据此丢弃、不回填
    assert done["version"] != client.get(f"/api/sessions/{sid}").json()["version"]


def test_check_unknown_session_404():
    assert client.post("/api/sessions/nope/check-stability").status_code == 404
