"""FastAPI 入口:设备维修记录 BM25 精确检索与症状证据简报。

- POST /documents/batch  批量增改删(语料、全库统计、剪枝依据一起生效)
- POST /search           正式检索(BlockMax WAND 块上界剪枝)
- POST /search/exhaustive 对照接口(全量计分后排序,仅用于验证一致性)
- POST /briefings        发布症状证据简报(仅保留最近一份)
- GET  /briefings/current 读取最近一份简报(含原文、出处、冲突、未覆盖词、过期状态)
- GET  /stats            当前状态统计
- GET  /healthz          健康检查
- GET  /docs             Swagger UI
"""
from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse

from . import briefing as briefing_mod
from .index import B, K1, MAX_DOCS, CapacityError, IndexManager, exhaustive_search, search
from .schemas import (
    BatchRequest,
    BatchResponse,
    BriefingRequest,
    BriefingResponse,
    BriefingSentence,
    Hit,
    SearchRequest,
    SearchResponse,
)


def create_app(manager: IndexManager | None = None) -> FastAPI:
    if manager is None:
        delay = float(os.environ.get("BM25_BUILD_DELAY", "0") or 0)
        manager = IndexManager(build_delay=delay)

    app = FastAPI(
        title="设备维修记录 BM25 检索与症状证据简报",
        version="1.1.0",
        description=(
            "精确 BM25(k1=1.2, b=0.75, IDF=ln(1+(N-df+0.5)/(df+0.5)))。"
            "正式检索使用 BlockMax WAND 块上界剪枝,前 k 与穷举一致,同分按编号升序。"
            "另提供症状证据简报:按 。!？ 切分原句,在至多 4 句、180 字符内"
            "选出覆盖症状词最多的原句;引用记录更正或删除后简报永久过期。"
        ),
    )
    app.state.manager = manager
    app.state.briefings = briefing_mod.BriefingStore(manager)

    @app.get("/", include_in_schema=False)
    def root():
        return RedirectResponse(url="/docs")

    @app.get("/healthz", tags=["meta"])
    def healthz():
        return {"status": "ok"}

    @app.get("/stats", tags=["meta"])
    def stats():
        s = manager.current()
        return {
            "docs": s.n,
            "avg_len": s.avgdl,
            "terms": len(s.terms),
            "k1": K1,
            "b": B,
            "max_docs": MAX_DOCS,
        }

    @app.post("/documents/batch", response_model=BatchResponse, tags=["corpus"])
    def batch_update(req: BatchRequest):
        """批量增改删。语料、全库统计与剪枝依据一起生效;失败保留旧状态。"""
        try:
            new_state, deleted = manager.apply_batch(
                [(d.id, d.text, d.tokens) for d in req.upserts],
                list(req.deletes),
            )
        except CapacityError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return BatchResponse(total=new_state.n, upserted=len(req.upserts), deleted=deleted)

    @app.post("/search", response_model=SearchResponse, tags=["search"])
    def search_ep(req: SearchRequest):
        """正式检索:块上界剪枝,返回命中编号、正文、分数及完整计分篇数。"""
        state = manager.current()  # 一次查询只用同一完整状态
        hits, scored = search(state, req.terms, req.k)
        return SearchResponse(
            hits=[Hit(id=i, text=t, score=sc) for i, t, sc in hits], scored=scored
        )

    @app.post("/search/exhaustive", response_model=SearchResponse, tags=["debug"])
    def search_exhaustive_ep(req: SearchRequest):
        """对照接口:全量计分后排序,仅用于验证正式检索的一致性,非正式路径。"""
        state = manager.current()
        hits, scored = exhaustive_search(state, req.terms, req.k)
        return SearchResponse(
            hits=[Hit(id=i, text=t, score=sc) for i, t, sc in hits], scored=scored
        )

    def _briefing_to_response(b) -> BriefingResponse:
        return BriefingResponse(
            published_at=b.published_at,
            expired=b.expired,
            record_ids=list(sorted(b.refs)),
            terms=list(b.terms),
            conflicts=list(b.conflicts),
            uncovered=list(b.uncovered),
            sentences=[
                BriefingSentence(record_id=rid, sentence_index=idx, sentence=text)
                for rid, idx, text in b.sentences
            ],
        )

    @app.post("/briefings", response_model=BriefingResponse, status_code=201, tags=["briefing"])
    def publish_briefing(req: BriefingRequest):
        """发布症状证据简报。

        校验失败(422)、无句可选或发布前引用记录已变化(409)均拒绝且不替换旧简报;
        成功后仅保留这一份。选句期间不持索引锁,不阻塞记录更正。
        """
        # 先固定一个不可变 State 引用;选句全程只用它,保证读到的原句与标注一致
        state = manager.current()
        try:
            selection = briefing_mod.validate_and_select(req, state)
            briefing = app.state.briefings.publish(req, state, selection)
        except briefing_mod.BriefingValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except (briefing_mod.NoSelectionError, briefing_mod.StaleReferenceError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return _briefing_to_response(briefing)

    @app.get("/briefings/current", response_model=BriefingResponse, tags=["briefing"])
    def get_briefing():
        """读取最近一份简报;过期仍返回发布时原文快照,经 expired 字段标识。无简报时 404。"""
        briefing = app.state.briefings.current()
        if briefing is None:
            raise HTTPException(status_code=404, detail="尚无已发布的简报")
        return _briefing_to_response(briefing)

    return app


app = create_app()
