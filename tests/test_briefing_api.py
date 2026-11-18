"""简报 API:发布校验、选句响应、只留最新一份、过期语义与并发拒绝。"""
import threading
import time

import pytest
from fastapi.testclient import TestClient

from app.index import IndexManager
from app.main import create_app


@pytest.fixture()
def client():
    return TestClient(create_app(IndexManager()))


def upsert(client, docs, deletes=None):
    return client.post(
        "/documents/batch",
        json={"upserts": docs, "deletes": deletes or []},
    )


def two_docs(client):
    upsert(client, [
        {"id": "R1", "text": "轴承过热明显,电机异响！未见漏油。", "tokens": ["轴承"]},
        {"id": "R2", "text": "复查确认轴承正常。", "tokens": ["轴承"]},
    ])


def publish(client, **over):
    payload = {
        "record_ids": ["R1", "R2"],
        "terms": ["轴承", "漏油"],
        "annotations": [
            {"record_id": "R1", "sentence_index": 0, "term": "轴承", "polarity": "positive"},
            {"record_id": "R1", "sentence_index": 1, "term": "漏油", "polarity": "negative"},
            {"record_id": "R2", "sentence_index": 0, "term": "轴承", "polarity": "negative"},
        ],
    }
    payload.update(over)
    return client.post("/briefings", json=payload)


def test_get_current_404_when_none(client):
    assert client.get("/briefings/current").status_code == 404


def test_publish_happy_path_fields(client):
    two_docs(client)
    r = publish(client)
    assert r.status_code == 201
    body = r.json()
    assert body["expired"] is False
    assert body["record_ids"] == ["R1", "R2"]
    assert body["terms"] == ["轴承", "漏油"]
    assert body["conflicts"] == ["轴承"]  # R1 肯定、R2 否定
    assert body["uncovered"] == []
    # 覆盖冲突词须各取一句 + 漏油一句,共 3 句,均 <=180 字符
    assert len(body["sentences"]) == 3
    assert sum(len(s["sentence"]) for s in body["sentences"]) <= 180
    sources = {(s["record_id"], s["sentence_index"]) for s in body["sentences"]}
    assert ("R1", 0) in sources and ("R1", 1) in sources and ("R2", 0) in sources
    assert body["sentences"] == sorted(body["sentences"], key=lambda s: (s["record_id"], s["sentence_index"]))
    assert "published_at" in body

    got = client.get("/briefings/current").json()
    assert got["sentences"] == body["sentences"]


def test_request_shape_bounds_422(client):
    two_docs(client)
    # 记录数 2~4
    r = publish(client, record_ids=["R1"])
    assert r.status_code == 422
    r = publish(client, record_ids=["R1", "R2", "R1"])  # 重复
    assert r.status_code == 422
    # 症状词 1~8
    assert publish(client, terms=[]).status_code == 422
    assert publish(client, terms=[f"w{i}" for i in range(9)]).status_code == 422
    # 标注 1~18
    assert publish(client, annotations=[]).status_code == 422
    anns = [{"record_id": "R1", "sentence_index": 0, "term": "轴承", "polarity": "positive"}] * 19
    assert publish(client, annotations=anns).status_code == 422
    # 极性枚举
    bad = [{"record_id": "R1", "sentence_index": 0, "term": "轴承", "polarity": "yes"}]
    assert publish(client, annotations=bad).status_code == 422


def test_semantic_validation_422_does_not_replace_old(client):
    two_docs(client)
    first = publish(client).json()
    # 编号不存在
    r = publish(client, record_ids=["R1", "NOPE"],
                annotations=[{"record_id": "R1", "sentence_index": 0,
                              "term": "轴承", "polarity": "positive"}])
    assert r.status_code == 422
    # 标注词不在原句
    r = publish(client, annotations=[
        {"record_id": "R1", "sentence_index": 0, "term": "漏油", "polarity": "positive"}
    ])
    assert r.status_code == 422 and "未出现" in r.json()["detail"]
    # 句序越界
    r = publish(client, annotations=[
        {"record_id": "R1", "sentence_index": 9, "term": "轴承", "polarity": "positive"}
    ])
    assert r.status_code == 422 and "越界" in r.json()["detail"]
    # 旧简报原样保留
    assert client.get("/briefings/current").json()["published_at"] == first["published_at"]


def test_no_selectable_sentence_409_keeps_old(client):
    upsert(client, [
        {"id": "R1", "text": "长" * 200 + "。", "tokens": []},
        {"id": "R2", "text": "短。", "tokens": []},
    ])
    ok = client.post("/briefings", json={
        "record_ids": ["R1", "R2"],
        "terms": ["短"],
        "annotations": [{"record_id": "R2", "sentence_index": 0, "term": "短", "polarity": "positive"}],
    })
    assert ok.status_code == 201
    before = ok.json()
    # 仅标注 201 字符的超长句 => 无句可选
    r = client.post("/briefings", json={
        "record_ids": ["R1", "R2"],
        "terms": ["长"],
        "annotations": [{"record_id": "R1", "sentence_index": 0, "term": "长", "polarity": "positive"}],
    })
    assert r.status_code == 409 and "无句可选" in r.json()["detail"]
    cur = client.get("/briefings/current").json()
    assert cur["published_at"] == before["published_at"]


def test_only_latest_briefing_kept(client):
    two_docs(client)
    publish(client)
    second = client.post("/briefings", json={
        "record_ids": ["R1", "R2"],
        "terms": ["漏油"],
        "annotations": [
            {"record_id": "R1", "sentence_index": 1, "term": "漏油", "polarity": "negative"}
        ],
    })
    assert second.status_code == 201
    cur = client.get("/briefings/current").json()
    assert cur["terms"] == ["漏油"]  # 旧的那份已被替换,只留最新


def test_correction_expires_and_snapshot_text_remains(client):
    two_docs(client)
    body = publish(client).json()
    original = body["sentences"][0]["sentence"]

    # 无关记录更正不影响
    upsert(client, [{"id": "R9", "text": "无关。", "tokens": []}])
    assert client.get("/briefings/current").json()["expired"] is False

    # 引用记录更正 => 过期
    upsert(client, [{"id": "R1", "text": "更正后的全新正文。", "tokens": []}])
    expired = client.get("/briefings/current").json()
    assert expired["expired"] is True
    # 仍显示发布时原文,不随更正变化
    assert expired["sentences"][0]["sentence"] == original
    assert all(s["record_id"] != "R1" or s["sentence"] != "更正后的全新正文。"
               for s in expired["sentences"])

    # 同号重录为原文 => 不恢复
    upsert(client, [{"id": "R1", "text": "轴承过热明显,电机异响！未见漏油。", "tokens": []}])
    assert client.get("/briefings/current").json()["expired"] is True

    # 删除引用记录同样过期(已过期则保持)
    upsert(client, [], deletes=["R2"])
    assert client.get("/briefings/current").json()["expired"] is True


def test_failed_batch_does_not_expire(client):
    two_docs(client)
    publish(client)
    # 构造容量超限的失败更正(422/409 之外,这里走 409 容量)
    from app.index import MAX_DOCS
    big = [{"id": f"X{i}", "text": "t", "tokens": ["x"]} for i in range(MAX_DOCS)]
    r = upsert(client, big)
    assert r.status_code == 409
    assert client.get("/briefings/current").json()["expired"] is False


def test_selection_does_not_block_correction():
    # 慢建索引占写锁期间,选句与读取仍即时返回(不等待写锁)。
    # 直接用核心对象构造真实并发:更正在持写锁构建,选句只读取不可变 State。
    mgr = IndexManager(build_delay=0.6)
    mgr.apply_batch([("R1", "轴承过热。", ()), ("R2", "复查正常。", ())], [])
    from app.briefing import BriefingStore

    store = BriefingStore(mgr)
    req = {
        "record_ids": ["R1", "R2"],
        "terms": ["轴承"],
        "annotations": [
            {"record_id": "R1", "sentence_index": 0, "term": "轴承", "polarity": "positive"}
        ],
    }

    class Ann:
        def __init__(self, d):
            self.record_id = d["record_id"]
            self.sentence_index = d["sentence_index"]
            self.term = d["term"]
            self.polarity = d["polarity"]

    class Req:
        def __init__(self, d):
            self.record_ids = d["record_ids"]
            self.terms = d["terms"]
            self.annotations = [Ann(a) for a in d["annotations"]]

    from app.briefing import validate_and_select

    done = []

    def updater():
        mgr.apply_batch([("R9", "无关更正。", ())], [])  # 持写锁约 0.6s
        done.append(True)

    t = threading.Thread(target=updater)
    t0 = time.monotonic()
    t.start()
    time.sleep(0.1)  # 更正进行中,写锁被占用
    state = mgr.current()
    sel = validate_and_select(Req(req), state)  # 不取写锁,立即返回
    assert sel.sentences[0].record_id == "R1"
    assert store.current() is None  # 读取简报同样不被阻塞
    assert time.monotonic() - t0 < 0.5
    t.join()
    assert done


def test_publish_rejected_when_refs_change_during_selection(client, monkeypatch):
    # 选句所用 State 在提交前已切换 => 409 且旧简报保留。
    # 让选句过程阻塞到更正提交完成,确定性地制造"发布前变化"窗口。
    # 注意:TestClient 的请求经同一 portal 串行,故更正在旁路线程直接走 manager。
    import app.briefing as briefing_mod

    mgr: IndexManager = client.app.state.manager
    two_docs(client)
    first = publish(client).json()
    entered = threading.Event()
    gate = threading.Event()
    real_validate = briefing_mod.validate_and_select

    def slow_validate(req, state):
        entered.set()       # 通知:选句已进入并持有旧 State
        gate.wait(2.0)      # 等待旁路更正完成提交切换
        return real_validate(req, state)

    monkeypatch.setattr(briefing_mod, "validate_and_select", slow_validate)

    def updater():
        entered.wait(2.0)   # 等选句进入后再更正,确保窗口确定
        mgr.apply_batch([("R9", "插入。", ())], [])
        gate.set()

    t = threading.Thread(target=updater)
    t.start()
    r = publish(client)
    t.join()
    assert r.status_code == 409 and "发布前已变化" in r.json()["detail"]
    cur = client.get("/briefings/current").json()
    assert cur["published_at"] == first["published_at"]  # 未替换旧简报


def test_search_still_works_alongside_briefings(client):
    two_docs(client)
    publish(client)
    r = client.post("/search", json={"terms": ["轴承"], "k": 5})
    assert r.status_code == 200
    assert {h["id"] for h in r.json()["hits"]} == {"R1", "R2"}
    assert client.get("/stats").json()["docs"] == 2


def test_uncovered_term_reported(client):
    upsert(client, [
        {"id": "R1", "text": "轴承过热。", "tokens": []},
        {"id": "R2", "text": "运行平稳。", "tokens": []},
    ])
    r = client.post("/briefings", json={
        "record_ids": ["R1", "R2"],
        "terms": ["轴承", "异响"],
        "annotations": [
            {"record_id": "R1", "sentence_index": 0, "term": "轴承", "polarity": "positive"}
        ],
    })
    assert r.status_code == 201
    body = r.json()
    assert body["uncovered"] == ["异响"]  # 无任何标注的词视为未覆盖
    assert body["conflicts"] == []
