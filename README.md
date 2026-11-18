# 设备维修记录 BM25 精确检索与症状证据简报

面向设备维护工程师的故障处理办法检索服务:FastAPI + NumPy 实现精确 BM25,
正式检索使用 BlockMax WAND 块上界剪枝(不全量计分后排序),前 k 与穷举完全一致。
另提供维修主管使用的症状证据简报:从现存维修记录中自动选出覆盖症状词最多的原句。

## 快速启动

```bash
docker compose up --build
```

- 交互式 API 文档: <http://localhost:8000/docs>
- 健康检查: <http://localhost:8000/healthz>

载入本地示例语料(1500 篇维修记录,含空文档):

```bash
python3 scripts/load_corpus.py            # 默认 http://localhost:8000
```

本地开发(不用 Docker):

```bash
pip install -r requirements-dev.txt
uvicorn app.main:app --port 8000
```

## 检索模型

- 固定 `k1 = 1.2`、`b = 0.75`;`IDF = ln(1 + (N − df + 0.5) / (df + 0.5))`(恒为正)。
- 文档得分 = Σ(去重查询词项) `idf · tf·(k1+1) / (tf + k1·(1 − b + b·dl/avgdl))`。
- 空文档(分词序列为空)计入 N 与平均长度 avgdl,不产生任何 posting,永远不会成为命中。
- 空库检索返回空结果(`{"hits": [], "scored": 0}`)。
- 同分按编号升序;查询词项服务端去重;`k` 限 1~20。

## 接口

### `POST /documents/batch` — 批量增改删

```json
{
  "upserts": [{"id": "R0001", "text": "数控机床轴承过热,更换轴承后恢复正常。", "tokens": ["数控机床", "轴承过热", "更换轴承"]}],
  "deletes": ["R0002"]
}
```

- 同一批内先删后增改;增改中重复编号后者覆盖前者。
- 语料、全库统计(N、avgdl、idf)与剪枝依据(块最大贡献)在一批中一起构建、原子生效。
- 更正后总篇数超过 5000 返回 `409`,旧状态完整保留;参数非法返回 `422`,旧状态不变。
- 写者串行:并发成功更正逐个生效、不丢失;读者不加锁:慢建索引期间查询仍用旧状态即时返回,
  构建成功后的新查询自动使用新状态(可用环境变量 `BM25_BUILD_DELAY=秒` 演示)。

### `POST /search` — 正式检索(块上界剪枝)

```json
{"terms": ["轴承过热", "更换轴承"], "k": 10}
```

响应:

```json
{
  "hits": [{"id": "R0001", "text": "...", "score": 3.42}],
  "scored": 88
}
```

- `hits`:命中记录的编号、正文、分数,按分数降序、同分编号升序,至多 k 篇。
- `scored`:完整计分篇数(通过枢轴检查、被完整累加分数的文档数)。
  块上界剪枝下 `scored` 通常远小于含词候选数与全库篇数——被跳过的候选未经完整计分。

### `POST /search/exhaustive` — 穷举对照(仅验证用)

全量计分后排序的对照接口,响应结构相同,`scored` 恒为全库篇数 N。
用于核对正式检索的前 k 一致性,不是正式检索路径。

### `POST /briefings` — 发布症状证据简报

维修主管提交引用记录、症状词与人工标注,服务自动选句并发布(仅保留最近一份)。

```json
{
  "record_ids": ["R0001", "R0002"],
  "terms": ["轴承", "漏油"],
  "annotations": [
    {"record_id": "R0001", "sentence_index": 0, "term": "轴承", "polarity": "positive"},
    {"record_id": "R0001", "sentence_index": 1, "term": "漏油", "polarity": "negative"},
    {"record_id": "R0002", "sentence_index": 0, "term": "轴承", "polarity": "negative"}
  ]
}
```

请求约束(违反返回 `422`,旧简报不受影响):

- `record_ids`:2~4 个**现存**记录编号,不可重复。
- `terms`:1~8 个症状词,不可重复。
- `annotations`:1~18 条人工标注;`sentence_index` 为按 `。!？` 切分后的原句序号(**从 0 起**),
  标注 `record_id` 必须在 `record_ids` 中、`term` 必须在 `terms` 中且出现在该原句里,
  序号越界同样 `422`。`polarity` 取 `positive`(肯定)/ `negative`(否定)。

选句规则:

- 原句按全角 `。!？` 切分,标点附在句末、跳过空段;只有被标注引用的原句才是候选。
- 选**至多 4 句**、**总字符数(含标点)不超过 180**,使覆盖的症状词最多。
- 同一词同时有肯定与否定标注(冲突词)时,须**各选一句**带对应极性的原句才算覆盖;
  未满足的词进入 `uncovered`。完全没有标注的症状词也算未覆盖。
- 覆盖数相同时先取总字符更少的;再相同则取 `(来源编号, 句序)` 升序序列字典序最小者。
- 候选句全部超 180 字符等导致**无句可选**时返回 `409`,本次拒绝且不替换旧简报。

响应(201)与 `GET /briefings/current` 同构:

```json
{
  "published_at": "2026-11-18T21:19:44.346535Z",
  "expired": false,
  "record_ids": ["R0001", "R0002"],
  "terms": ["轴承", "漏油"],
  "conflicts": ["轴承"],
  "uncovered": [],
  "sentences": [
    {"record_id": "R0001", "sentence_index": 0, "sentence": "轴承过热明显,电机异响！"}
  ]
}
```

### `GET /briefings/current` — 读取最近一份简报

- 只保留最近发布的一份;从未发布返回 `404`。
- `sentences` 给出原句快照、出处(`record_id` + `sentence_index`);`conflicts` 为肯否冲突词,
  `uncovered` 为未完全覆盖的词,`expired` 为过期状态。
- **过期规则**:任一引用记录被更正或删除,简报即置为过期且**永久不可恢复**——
  同号重录(即使内容改回原文)也不恢复;更正无关记录不影响;容量超限等失败更正不产生过期。
- 过期后读取仍返回**发布时的原句快照**,正文不随后续更正变化,仅以 `expired: true` 标识。
- 选句过程不持索引写锁,不阻塞记录更正;若引用记录在发布前已变化,
  本次发布返回 `409` 且不替换旧简报(乐观并发,基于不可变 State 引用判定)。

### 其他

- `GET /stats`:篇数、平均长度、词项数、BM25 参数。
- `GET /healthz`:健康检查。

## 剪枝正确性说明

每个词项的 posting 按 32 篇分块,预计算块内最大贡献;块最大贡献的后缀最大值即
"当前位置之后所有 posting 贡献"的上界。WAND 每轮按当前文档号排序游标,以前缀上界和
选择枢轴:上界和严格小于当前第 k 名分数的文档不可能进入前 k,按块整块跳过;
只有与枢轴文档对齐的候选才被完整计分。枢轴判断取 `>=` 阈值,同分文档仍可凭编号升序
进入前 k,保证与穷举逐位一致(同一 State、同一词项顺序累加,浮点结果完全相同)。

## 验证

```bash
python3 -m pytest tests/ -q     # 单元/API/并发测试(含简报选句与过期语义)
python3 scripts/verify.py       # 穷举对照 + 随机回归 + 剪枝示例
```

`scripts/verify.py` 用本地语料 `data/corpus.json` 与随机语料,将 BlockMax WAND 与
穷举实现、独立参照实现(直接按公式对原始语料计分)逐位比对,并打印完整计分篇数,
展示确有候选被跳过,例如:

```
查询 ['报警', '停机'], k=10:完整计分 88 篇;含查询词的文档 449 篇;全库 1500 篇
即 361 篇含词候选未经完整计分即被块上界跳过,前 10 名与穷举完全一致。
```

## 目录

```
app/briefing.py  症状证据简报:原句切分、标注校验、选句优化、过期与并发控制
app/index.py      BM25 状态构建、BlockMax WAND、穷举对照、原子状态管理、提交回调
app/main.py       FastAPI 应用与端点(检索 + 简报)
app/schemas.py    请求/响应模型
data/corpus.json  本地示例语料(scripts/gen_corpus.py 生成,确定性种子)
scripts/verify.py 穷举对照与剪枝效果验证
scripts/load_corpus.py 向运行中的服务批量写入语料
tests/            单元、API 与并发一致性测试(含简报)
```
