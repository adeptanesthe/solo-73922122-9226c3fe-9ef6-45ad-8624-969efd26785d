"""简报选句核心:切句、覆盖/冲突规则、字符上限、字典序兜底与过期判定。"""
import threading
import time

import pytest

from app.briefing import (
    BriefingStore,
    BriefingValidationError,
    NoSelectionError,
    StaleReferenceError,
    split_sentences,
    validate_and_select,
)
from app.index import MAX_DOCS, CapacityError, IndexManager


class Req:
    """轻量请求替身,字段与 schemas.BriefingRequest 一致。"""

    def __init__(self, record_ids, terms, annotations):
        self.record_ids = record_ids
        self.terms = terms
        self.annotations = [Ann(*a) for a in annotations]


class Ann:
    def __init__(self, record_id, sentence_index, term, polarity):
        self.record_id = record_id
        self.sentence_index = sentence_index
        self.term = term
        self.polarity = polarity


def manager_with(docs):
    mgr = IndexManager()
    mgr.apply_batch(
        [(did, text, ()) for did, text in docs.items()], []
    )
    return mgr


def select(mgr, record_ids, terms, annotations):
    return validate_and_select(Req(record_ids, terms, annotations), mgr.current())


# ------------------------------ 切句 ------------------------------

def test_split_sentences_on_three_marks():
    sents = split_sentences("轴承过热。电机异响！停机？已处理")
    assert sents == ["轴承过热。", "电机异响！", "停机？", "已处理"]


def test_split_sentences_keeps_punct_and_skips_empty():
    # 开头/连续标点不产生空句;标点附在所属句末
    assert split_sentences("。！前句。。尾句？") == ["前句。。", "尾句？"]
    assert split_sentences("无标点整段") == ["无标点整段"]
    assert split_sentences("") == []
    assert split_sentences("。！？") == []
    # 半角 ! ? 不是切分符,仅全角 。!？ 切分
    assert split_sentences("电机异响!停机?观察") == ["电机异响!停机?观察"]


# ------------------------------ 校验 ------------------------------

def test_missing_record_rejected():
    mgr = manager_with({"A": "甲句。", "B": "乙句。"})
    with pytest.raises(BriefingValidationError, match="不存在"):
        select(mgr, ["A", "X"], ["甲"], [("A", 0, "甲", "positive")])


def test_duplicate_ids_or_terms_rejected():
    mgr = manager_with({"A": "甲句。", "B": "乙句。"})
    with pytest.raises(BriefingValidationError, match="不可重复"):
        select(mgr, ["A", "A"], ["甲"], [("A", 0, "甲", "positive")])
    with pytest.raises(BriefingValidationError, match="不可重复"):
        select(mgr, ["A", "B"], ["甲", "甲"], [("A", 0, "甲", "positive")])


def test_annotation_term_must_be_in_terms():
    mgr = manager_with({"A": "甲句。", "B": "乙句。"})
    with pytest.raises(BriefingValidationError, match="不在 terms"):
        select(mgr, ["A", "B"], ["甲"], [("A", 0, "句", "positive")])


def test_annotation_record_must_be_in_refs():
    mgr = manager_with({"A": "甲句。", "B": "乙句。"})
    with pytest.raises(BriefingValidationError, match="不在 record_ids"):
        select(mgr, ["A"], ["乙"], [("B", 0, "乙", "positive")])


def test_sentence_index_out_of_range():
    mgr = manager_with({"A": "只有一句。", "B": "乙。"})
    with pytest.raises(BriefingValidationError, match="越界"):
        select(mgr, ["A", "B"], ["只"], [("A", 1, "只", "positive")])


def test_term_not_in_sentence_rejected():
    mgr = manager_with({"A": "轴承过热。", "B": "电机异响。"})
    with pytest.raises(BriefingValidationError, match="未出现"):
        select(mgr, ["A", "B"], ["电机"], [("A", 0, "电机", "positive")])


# ------------------------------ 选句规则 ------------------------------

def test_max_coverage_prefers_more_terms():
    # A 句覆盖两词,B 句覆盖第三词;三词全覆盖需两句
    mgr = manager_with({"A": "轴承过热且电机异响。", "B": "单独停机。"})
    sel = select(
        mgr, ["A", "B"], ["轴承", "电机", "停机"],
        [("A", 0, "轴承", "positive"),
         ("A", 0, "电机", "positive"),
         ("B", 0, "停机", "positive")],
    )
    assert len(sel.sentences) == 2
    assert sel.uncovered == ()


def test_at_most_four_sentences():
    docs = {f"D{i}": f"症状{i}号。" for i in range(5)}
    mgr = manager_with(docs)
    anns = [(f"D{i}", 0, f"症状{i}", "positive") for i in range(5)]
    terms = [f"症状{i}" for i in range(5)]
    sel = select(mgr, [f"D{i}" for i in range(4)], terms[:4], anns[:4])
    assert len(sel.sentences) == 4


def test_total_char_limit_drops_otherwise_needed_sentence():
    long_text = "长" * 179 + "。"  # 180 字符,恰好等于上限,可选
    over_text = "长" * 180 + "。"  # 181 字符,单句即超限
    mgr = manager_with({"A": long_text, "B": over_text})
    sel = select(
        mgr, ["A", "B"], ["长"],
        [("A", 0, "长", "positive"), ("B", 0, "长", "positive")],
    )
    picked = [(c.record_id, c.sentence_index) for c in sel.sentences]
    assert picked == [("A", 0)]  # B 句 181 字符不可选


def test_no_selectable_sentence_raises():
    mgr = manager_with({"A": "长" * 181 + "。", "B": "短。"})
    # 只标注超长句 -> 无句可选
    with pytest.raises(NoSelectionError):
        select(mgr, ["A", "B"], ["长"], [("A", 0, "长", "positive")])


def test_180_char_boundary_includes_punctuation():
    # 179 汉字 + 句号 = 180 字符(含标点),恰好可选
    mgr = manager_with({"A": "长" * 179 + "。", "B": "短。"})
    sel = select(mgr, ["A", "B"], ["长"], [("A", 0, "长", "positive")])
    assert sel.sentences[0].record_id == "A"
    assert len(sel.sentences[0].sentence) == 180
    # 180 汉字 + 句号 = 181,超上限不可选
    mgr2 = manager_with({"A": "长" * 180 + "。", "B": "短。"})
    with pytest.raises(NoSelectionError):
        select(mgr2, ["A", "B"], ["长"], [("A", 0, "长", "positive")])


def test_combined_chars_must_fit_180():
    mgr = manager_with({
        "A": "甲" * 100 + "。",   # 101
        "B": "乙" * 100 + "。",   # 101
    })
    sel = select(
        mgr, ["A", "B"], ["甲", "乙"],
        [("A", 0, "甲", "positive"), ("B", 0, "乙", "positive")],
    )
    # 两句合计 202 > 180,只能各覆盖其一;等字符时按字典序取 A,乙未覆盖
    assert len(sel.sentences) == 1
    assert sel.sentences[0].record_id == "A"
    assert sel.uncovered == ("乙",)


def test_conflict_requires_both_polarities_each_with_a_sentence():
    mgr = manager_with({
        "A": "轴承过热明显。",
        "B": "未见轴承过热。",
    })
    # 同词肯否两标,但否定句之外没有第二句承载否定 -> 必须两句都选才覆盖
    sel = select(
        mgr, ["A", "B"], ["轴承"],
        [("A", 0, "轴承", "positive"),
         ("B", 0, "轴承", "negative")],
    )
    assert sel.conflicts == ("轴承",)
    assert sel.uncovered == ()
    assert [c.record_id for c in sel.sentences] == ["A", "B"]


def test_conflict_missing_one_polarity_is_uncovered():
    mgr = manager_with({"A": "轴承过热明显。", "B": "其他现象。"})
    sel = select(
        mgr, ["A", "B"], ["轴承"],
        [("A", 0, "轴承", "positive"),
         ("A", 0, "轴承", "negative")],  # 否定也只在 A 句
    )
    # 同一句同时带肯否两标,两个极性都有句承载,视为覆盖
    assert sel.uncovered == ()
    sel2 = select(
        mgr, ["A", "B"], ["轴承"],
        [("A", 0, "轴承", "positive")],  # 只有肯定,而需求要求看冲突:无否定即无冲突
    )
    assert sel2.conflicts == () and sel2.uncovered == ()


def test_tie_break_fewer_chars_then_lexicographic():
    # 同覆盖 1 词:短句(字符少)优先
    mgr = manager_with({"A": "短甲。", "B": "很长很长的甲。"})
    sel = select(
        mgr, ["A", "B"], ["甲"],
        [("A", 0, "甲", "positive"), ("B", 0, "甲", "positive")],
    )
    assert sel.sentences[0].record_id == "A"

    # 等覆盖等字符:来源编号+句序升序序列字典序最小
    mgr2 = manager_with({"A": "甲字。", "B": "甲字。"})
    sel2 = select(
        mgr2, ["A", "B"], ["甲"],
        [("A", 0, "甲", "positive"), ("B", 0, "甲", "positive")],
    )
    assert sel2.sentences[0].record_id == "A"


def test_lexicographic_sequence_compares_full_tuple():
    # 两句组合 vs 一句组合覆盖相同时,字符少者优先;等字符时比较完整序列
    mgr = manager_with({
        "A": "甲。",          # 2 字符
        "B": "乙。",          # 2 字符
        "C": "甲乙共句。",    # 5 字符,一句覆盖两词
    })
    sel = select(
        mgr, ["A", "B", "C"], ["甲", "乙"],
        [("A", 0, "甲", "positive"),
         ("B", 0, "乙", "positive"),
         ("C", 0, "甲", "positive"),
         ("C", 0, "乙", "positive")],
    )
    # A+B 共 4 字符覆盖两词;C 单句 5 字符也覆盖两词 -> 取字符少的 A+B
    assert [c.record_id for c in sel.sentences] == ["A", "B"]


def test_duplicate_annotations_on_same_sentence_merge():
    mgr = manager_with({"A": "轴承过热。", "B": "电机异响。"})
    sel = select(
        mgr, ["A", "B"], ["轴承"],
        [("A", 0, "轴承", "positive"),
         ("A", 0, "轴承", "positive")],  # 重复标注不是错误,合并为同一候选
    )
    assert len(sel.sentences) == 1 and sel.sentences[0].record_id == "A"


# ------------------------------ 过期与并发 ------------------------------

def _publish(mgr, refs=("A", "B")):
    store = BriefingStore(mgr)
    req = Req(list(refs), ["甲"], [("A", 0, "甲", "positive")])
    sel = validate_and_select(req, mgr.current())
    return store, store.publish(req, mgr.current(), sel)


def test_correction_expires_briefing_and_rerecord_keeps_expired():
    mgr = manager_with({"A": "甲句。", "B": "乙句。"})
    store, b = _publish(mgr)
    assert b.expired is False
    mgr.apply_batch([("A", "甲句改。", ())], [])  # 引用记录更正
    assert store.current().expired is True
    mgr.apply_batch([("A", "甲句。", ())], [])  # 同号重录为原文,不恢复
    assert store.current().expired is True


def test_delete_expires_but_unrelated_correction_keeps():
    mgr = manager_with({"A": "甲句。", "B": "乙句。", "C": "丙句。"})
    store, _ = _publish(mgr)
    mgr.apply_batch([("C", "丙句改。", ())], [])  # 无关记录更正
    assert store.current().expired is False
    mgr.apply_batch([], ["B"])  # 删除引用记录
    assert store.current().expired is True
    assert store.current().sentences[0][2] == "甲句。"  # 原文快照仍在


def test_failed_correction_does_not_expire():
    mgr = manager_with({"A": "甲句。", "B": "乙句。"})
    store, _ = _publish(mgr)
    big = [(f"D{i}", "t", ("x",)) for i in range(MAX_DOCS)]
    with pytest.raises(CapacityError):
        mgr.apply_batch(big, [])
    assert store.current().expired is False


def test_stale_state_publish_rejected_and_keeps_old():
    mgr = manager_with({"A": "甲句。", "B": "乙句。", "C": "丙句。"})
    store, old = _publish(mgr)
    stale_state = mgr.current()
    mgr.apply_batch([("C", "丙句改。", ())], [])  # 索引已切换(无关记录也换了 State)
    req = Req(["A", "C"], ["丙"], [("C", 0, "丙", "positive")])
    with pytest.raises(StaleReferenceError):
        store.publish(req, stale_state, validate_and_select(req, stale_state))
    assert store.current() is old  # 旧简报未被替换


def test_slow_batch_does_not_block_selection_or_reads():
    mgr = IndexManager(build_delay=0.6)
    mgr.apply_batch([("A", "甲句。", ()), ("B", "乙句。", ())], [])
    store, _ = _publish(mgr)

    done = []

    def updater():
        mgr.apply_batch([("C", "丙句。", ())], [])
        done.append(True)

    t = threading.Thread(target=updater)
    t0 = time.monotonic()
    t.start()
    time.sleep(0.1)  # 更正进行中(占写锁)
    req = Req(["A", "B"], ["甲"], [("A", 0, "甲", "positive")])
    sel = validate_and_select(req, mgr.current())  # 选句不被写锁阻塞
    assert sel.sentences[0].record_id == "A"
    assert store.current() is not None  # 读取也不被阻塞
    assert time.monotonic() - t0 < 0.5
    t.join()
    assert done and store.current().expired is False  # C 与引用无关
