"""RRF 融合：去重、名次加权、截断。

线上这条链路直接影响"两路召回谁说了算"，但原来只靠 print 肉眼观察。
"""

from __future__ import annotations

from knowledge.processor.query_process.config import QueryConfig
from knowledge.processor.query_process.nodes.rrf import RrfNode


def make_node(k: int = 60, max_results: int = 10) -> RrfNode:
    config = QueryConfig()
    config.rrf_k = k
    config.rrf_max_results = max_results
    return RrfNode(config=config)


def entity(chunk_id: str, content: str = "") -> dict:
    return {"chunk_id": chunk_id, "content": content or chunk_id, "title": chunk_id}


def test_doc_hitting_both_lists_ranks_first():
    node = make_node()
    dense = [entity("a"), entity("b"), entity("c")]
    hyde = [entity("b"), entity("a")]

    fused = node._rrf([(dense, 1.0), (hyde, 1.0)], rrf_k=60, rrf_max_results=10)
    ids = [item["chunk_id"] for item, _ in fused]

    # a: 1/61 + 1/62，b: 1/62 + 1/61 —— 两者相等，但都高于只在一路出现的 c
    assert set(ids[:2]) == {"a", "b"}
    assert ids[-1] == "c"


def test_scores_follow_rrf_formula():
    node = make_node()
    fused = node._rrf([([entity("x")], 1.0)], rrf_k=60, rrf_max_results=5)
    score = fused[0][1]
    assert score == 1.0 / 61


def test_weight_is_applied():
    node = make_node()
    fused = node._rrf([([entity("x")], 2.0)], rrf_k=60, rrf_max_results=5)
    assert fused[0][1] == 2.0 / 61


def test_duplicate_across_lists_keeps_single_entry():
    node = make_node()
    fused = node._rrf(
        [([entity("a"), entity("a")], 1.0), ([entity("a")], 1.0)],
        rrf_k=60,
        rrf_max_results=10,
    )
    assert len([f for f in fused if f[0]["chunk_id"] == "a"]) == 1


def test_missing_chunk_id_is_skipped():
    node = make_node()
    fused = node._rrf(
        [([{"content": "没有 id"}, entity("ok")], 1.0)],
        rrf_k=60,
        rrf_max_results=10,
    )
    assert [item["chunk_id"] for item, _ in fused] == ["ok"]


def test_max_results_truncates():
    node = make_node()
    dense = [entity(f"c{i}") for i in range(20)]
    fused = node._rrf([(dense, 1.0)], rrf_k=60, rrf_max_results=5)
    assert len(fused) == 5


def test_empty_inputs_return_empty():
    node = make_node()
    assert node._rrf([([], 1.0), ([], 1.0)], rrf_k=60, rrf_max_results=10) == []