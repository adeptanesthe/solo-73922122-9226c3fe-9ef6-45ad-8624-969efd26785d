"""请求/响应模型。"""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

Polarity = Literal["positive", "negative"]


class DocUpsert(BaseModel):
    """单篇记录的增改:编号、正文、分词序列。"""

    id: str = Field(min_length=1, max_length=128, description="记录编号")
    text: str = Field(default="", description="记录正文")
    tokens: list[str] = Field(default_factory=list, description="分词序列(可为空,空文档计入统计)")


class BatchRequest(BaseModel):
    """批量增改删:同一批内先删后增改。"""

    upserts: list[DocUpsert] = Field(default_factory=list, description="新增/修改的记录")
    deletes: list[str] = Field(default_factory=list, description="删除的编号")


class BatchResponse(BaseModel):
    status: str = "ok"
    total: int = Field(description="生效后的总篇数")
    upserted: int = Field(description="本批提交的增改条数")
    deleted: int = Field(description="本批实际删除的篇数")


class SearchRequest(BaseModel):
    terms: list[str] = Field(default_factory=list, description="查询词项(服务端再去重)")
    k: int = Field(ge=1, le=20, description="返回篇数,1~20")


class Hit(BaseModel):
    id: str
    text: str
    score: float


class SearchResponse(BaseModel):
    hits: list[Hit]
    scored: int = Field(description="完整计分篇数")


class Annotation(BaseModel):
    """一条人工标注:某记录的某条原句中,某症状词是肯定还是否定证据。"""

    record_id: str = Field(min_length=1, max_length=128, description="现存记录编号(须在 record_ids 中)")
    sentence_index: int = Field(ge=0, description="原句序号,按 。!？ 切分后从 0 起的下标")
    term: str = Field(min_length=1, max_length=128, description="症状词(须在 terms 中且出现在该原句里)")
    polarity: Polarity = Field(description="positive=肯定证据,negative=否定证据")


class BriefingRequest(BaseModel):
    """发布症状证据简报。"""

    record_ids: list[str] = Field(
        min_length=2, max_length=4, description="2~4 个现存记录编号(不可重复)"
    )
    terms: list[str] = Field(
        min_length=1, max_length=8, description="1~8 个症状词(不可重复)"
    )
    annotations: list[Annotation] = Field(
        min_length=1, max_length=18, description="1~18 条人工标注"
    )


class BriefingSentence(BaseModel):
    record_id: str = Field(description="来源记录编号")
    sentence_index: int = Field(description="该记录内的原句序号(从 0 起)")
    sentence: str = Field(description="发布时截取的原句快照")


class BriefingResponse(BaseModel):
    published_at: datetime = Field(description="发布时间(UTC,ISO 8601)")
    expired: bool = Field(description="引用记录是否已被更正/删除而过期")
    record_ids: list[str] = Field(description="发布时引用的记录编号")
    terms: list[str] = Field(description="发布时提交的症状词")
    conflicts: list[str] = Field(description="同一词同时存在肯/否标注的冲突词")
    uncovered: list[str] = Field(description="未能完全覆盖(冲突词缺任一极性)的症状词")
    sentences: list[BriefingSentence] = Field(
        max_length=4, description="选中的原句(至多 4 句,总字符 <=180 含标点)"
    )
