"""症状证据简报:原句切分、人工标注校验、选句优化与过期管理。

选句规则(与需求逐条对应):
- 原句按 。!？ 切分,标点附在句末,跳过空段;句序号为切分结果下标(从 0 起)。
- 候选句 = 人工标注所引用的(记录, 句序)去重集合;标注词必须出现在所标原句中。
- 至多选择 4 句、总字符数(含标点)不超过 180,使覆盖的症状词最多。
- 同一词同时存在肯定与否定标注时,须各有一句带对应极性标注才算覆盖。
- 同覆盖数先取总字符少的,再取 (来源编号, 句序) 升序序列字典序最小者。
- 没有任何可行选句(候选均超长等)时抛 NoSelectionError,由端点拒绝且不替换旧简报。

过期管理:
- BriefingStore 订阅 IndexManager 的提交事件;引用记录被更正或删除即永久过期,
  同号重录不恢复;无关记录更正不影响;失败更正(容量超限等)不触发提交、不过期。
- 发布前捕获 State 引用,选句不持索引锁(不阻塞更正);提交时若 State 已变化则拒绝。
"""
from __future__ import annotations

import itertools
import re
import threading
from dataclasses import dataclass, replace
from datetime import datetime, timezone

MAX_SENTENCES = 4
MAX_TOTAL_CHARS = 180

_SENTENCE_RE = re.compile(r"[^。！？]+[。！？]*")


def split_sentences(text: str) -> list[str]:
    """按 。!？ 切分原句。

    标点附在所属句末;连续标点归于前句;跳过空段(含开头/连续标点产生的空段),
    因此句序号始终对应一条非空原句。无结句标点的尾部文字也独立成句。
    """
    return _SENTENCE_RE.findall(text)


class BriefingValidationError(ValueError):
    """请求语义不合法(编号不存在、越界、标注词不在原句等),对应 HTTP 422。"""


class NoSelectionError(RuntimeError):
    """无满足字符数限制的可选原句,对应 HTTP 409,旧简报保留。"""


class StaleReferenceError(RuntimeError):
    """选句期间引用记录发生更正,本次发布拒绝且不替换旧简报,对应 HTTP 409。"""


class _LexSeq:
    """(来源编号, 句序) 升序序列,按字典序比较。

    与 Python tuple 的区别:一个序列是另一个的前缀时,较短者更小(而非按长度外推),
    使"字典序最小"在不同句数的组合间仍有确定含义。
    """

    __slots__ = ("seq",)

    def __init__(self, seq: tuple):
        self.seq = seq

    def __lt__(self, other: "_LexSeq") -> bool:
        for a, b in zip(self.seq, other.seq):
            if a != b:
                return a < b
        return len(self.seq) < len(other.seq)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _LexSeq) and self.seq == other.seq

    def __le__(self, other: "_LexSeq") -> bool:
        return self == other or self < other

    def __gt__(self, other: "_LexSeq") -> bool:
        return other < self

    def __ge__(self, other: "_LexSeq") -> bool:
        return self == other or other < self

    def __hash__(self) -> int:
        return hash(self.seq)


class _ReverseLexSeq:
    """_LexSeq 的反向序:使"字典序最小"在 max 式选优中胜出。"""

    __slots__ = ("inner",)

    def __init__(self, inner: "_LexSeq"):
        self.inner = inner

    def __lt__(self, other: "_ReverseLexSeq") -> bool:
        return self.inner > other.inner

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _ReverseLexSeq) and self.inner == other.inner

    def __gt__(self, other: "_ReverseLexSeq") -> bool:
        return self.inner < other.inner

    def __le__(self, other: "_ReverseLexSeq") -> bool:
        return self == other or self < other

    def __ge__(self, other: "_ReverseLexSeq") -> bool:
        return self == other or other < self

    def __hash__(self) -> int:
        return hash(self.inner)


@dataclass(frozen=True)
class Candidate:
    """一条候选原句及其上的 (症状词, 极性) 标注集合。"""

    record_id: str
    sentence_index: int
    sentence: str
    tags: frozenset  # frozenset[(term, polarity)]


@dataclass(frozen=True)
class Selection:
    """选句结果。"""

    sentences: tuple  # tuple[Candidate, ...],按 (record_id, sentence_index) 升序
    conflicts: tuple  # tuple[str, ...]:同时有肯/否标注的症状词(按提交词序)
    uncovered: tuple  # tuple[str, ...]:未能完全覆盖的症状词(按提交词序)


@dataclass(frozen=True)
class StoredBriefing:
    """已发布简报(原文快照)。expired 仅通过 replace 切换,发布后内容不可变。"""

    published_at: datetime
    refs: frozenset  # frozenset[str]:引用的记录编号
    terms: tuple  # tuple[str, ...]:提交的症状词(保序)
    conflicts: tuple
    uncovered: tuple
    sentences: tuple  # tuple[(record_id, sentence_index, sentence), ...]
    expired: bool = False


def validate_and_select(req, state) -> Selection:
    """依据给定 State 校验请求并完成选句;所有判断只使用该不可变 State。"""
    ids = list(req.record_ids)
    id_set = set(ids)
    terms = list(req.terms)
    term_set = set(terms)

    if len(id_set) != len(ids):
        raise BriefingValidationError("record_ids 不可重复")
    if any(not rid.strip() for rid in ids):
        raise BriefingValidationError("record_ids 不可为空字符串")
    if len(term_set) != len(terms):
        raise BriefingValidationError("terms 不可重复")
    if any(not t.strip() for t in terms):
        raise BriefingValidationError("症状词不可为空字符串")

    missing = [rid for rid in ids if rid not in state.docs]
    if missing:
        raise BriefingValidationError(f"记录编号不存在: {missing}")

    sentences_of = {rid: split_sentences(state.docs[rid][0]) for rid in ids}

    # 标注收集:(record_id, sentence_index) -> [sentence, {(term, polarity)}]
    raw: dict = {}
    required: dict = {}  # term -> {polarity}
    for ann in req.annotations:
        if ann.record_id not in id_set:
            raise BriefingValidationError(
                f"标注引用的记录 {ann.record_id!r} 不在 record_ids 中"
            )
        if ann.term not in term_set:
            raise BriefingValidationError(
                f"标注症状词 {ann.term!r} 不在 terms 中"
            )
        sents = sentences_of[ann.record_id]
        if not 0 <= ann.sentence_index < len(sents):
            raise BriefingValidationError(
                f"记录 {ann.record_id} 原句序号 {ann.sentence_index} 越界"
                f"(共 {len(sents)} 句,序号从 0 起)"
            )
        sentence = sents[ann.sentence_index]
        if ann.term not in sentence:
            raise BriefingValidationError(
                f"标注词 {ann.term!r} 未出现在记录 {ann.record_id}"
                f"第 {ann.sentence_index} 句原句中"
            )
        key = (ann.record_id, ann.sentence_index)
        entry = raw.get(key)
        if entry is None:
            raw[key] = [sentence, {(ann.term, ann.polarity)}]
        else:
            entry[1].add((ann.term, ann.polarity))
        required.setdefault(ann.term, set()).add(ann.polarity)

    candidates = [
        Candidate(rid, idx, sentence, frozenset(tags))
        for (rid, idx), (sentence, tags) in sorted(raw.items())
    ]
    feasible = [c for c in candidates if len(c.sentence) <= MAX_TOTAL_CHARS]
    if not feasible:
        raise NoSelectionError(
            f"无句可选:{len(candidates)} 条候选原句均超过 {MAX_TOTAL_CHARS} 字符上限,"
            "本次拒绝且不替换旧简报"
        )

    # 枚举至多 4 句的全部子集(候选 <= 18,组合数 <= 4047),按规则取全局最优
    best_key = None  # (覆盖数, -总字符数, 升序序列),越大越优
    best_combo: tuple = ()
    best_tags: frozenset = frozenset()
    upper = min(MAX_SENTENCES, len(feasible))
    for size in range(1, upper + 1):
        for combo in itertools.combinations(feasible, size):
            total_chars = sum(len(c.sentence) for c in combo)
            if total_chars > MAX_TOTAL_CHARS:
                continue
            tags: set = set()
            for c in combo:
                tags |= c.tags
            coverage = sum(
                1
                for t in terms
                if all((t, p) in tags for p in required.get(t, ()))
            )
            # 最优规则:覆盖数最大、总字符最小、序列字典序最小。
            # key 统一取"越大越优",故后两项取反(序列用反向比较包装)。
            seq = _LexSeq(tuple((c.record_id, c.sentence_index) for c in combo))
            key = (coverage, -total_chars, _ReverseLexSeq(seq))
            if best_key is None or key > best_key:
                best_key = key
                best_combo = combo
                best_tags = frozenset(tags)

    conflicts = tuple(
        t for t in terms if required.get(t) == {"positive", "negative"}
    )
    uncovered = tuple(
        t
        for t in terms
        if t not in required
        or not all((t, p) in best_tags for p in required[t])
    )
    return Selection(sentences=best_combo, conflicts=conflicts, uncovered=uncovered)


class BriefingStore:
    """持有最近发布的一份简报;写者随索引提交事件标记过期,读者无锁阻塞之虑。

    锁序固定为 IndexManager 写锁 -> BriefingStore 锁;publish 只在最后提交时取本锁,
    选句过程不持任何锁,因此选句不阻塞更正。
    """

    def __init__(self, manager):
        self._manager = manager
        self._lock = threading.Lock()
        self._briefing: StoredBriefing | None = None
        manager.add_commit_listener(self.on_commit)

    def publish(self, req, state: object, selection: Selection) -> StoredBriefing:
        """提交新简报。state 必须是选句时使用的同一 State 引用。"""
        with self._lock:
            # 乐观并发检查:选句期间若索引已提交切换,本次拒绝且不替换旧简报
            if self._manager.current() is not state:
                raise StaleReferenceError(
                    "引用记录在本次发布前已变化,本次拒绝且不替换旧简报"
                )
            briefing = StoredBriefing(
                published_at=datetime.now(timezone.utc),
                refs=frozenset(req.record_ids),
                terms=tuple(req.terms),
                conflicts=selection.conflicts,
                uncovered=selection.uncovered,
                sentences=tuple(
                    (c.record_id, c.sentence_index, c.sentence)
                    for c in selection.sentences
                ),
            )
            self._briefing = briefing  # 仅保留最近一份
            return briefing

    def on_commit(self, changed_ids: set) -> None:
        """索引提交回调(在 IndexManager 写锁内调用):引用编号被触及即永久过期。"""
        with self._lock:
            b = self._briefing
            if b is not None and not b.expired and b.refs.intersection(changed_ids):
                self._briefing = replace(b, expired=True)

    def current(self) -> StoredBriefing | None:
        with self._lock:
            return self._briefing
