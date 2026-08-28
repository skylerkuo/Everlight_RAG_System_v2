from rag_app.qa.context import _answer_context
from rag_app.retrieval.bge_m3_index import BGEM3Index, SearchResult


class DummySettings:
    pass


def _chunk(doc: str, idx: int, text: str, kind: str = "html") -> dict:
    return {
        "chunk_id": f"{doc}-{idx}",
        "chunk_index": idx,
        "document_id": doc,
        "source_kind": kind,
        "title": doc,
        "source_url": f"https://example.test/{doc}",
        "heading_path": ["Section"],
        "content": text,
    }


def test_neighbor_lookup_stays_within_same_document():
    index = BGEM3Index(DummySettings(), load_model=False)
    a1 = _chunk("docA", 1, "A1")
    a2 = _chunk("docA", 2, "A2")
    a3 = _chunk("docA", 3, "A3")
    b1 = _chunk("docB", 1, "B1")
    index.chunks = [a1, a2, a3, b1]

    previous, following = index.get_chunk_neighbors(a2, radius=1)

    assert previous == [a1]
    assert following == [a3]
    assert b1 not in previous + following


def test_final_answer_context_adds_previous_and_next_text_without_new_source_labels():
    index = BGEM3Index(DummySettings(), load_model=False)
    a1 = _chunk("docA", 1, "previous product text")
    a2 = _chunk("docA", 2, "main product text")
    a3 = _chunk("docA", 3, "next product features")
    index.chunks = [a1, a2, a3]

    result = SearchResult(rank=1, score=0.9, chunk=a2)
    context, audit = _answer_context([result], index, neighbor_radius=1)

    assert "[S1]" in context
    assert "[S2]" not in context
    assert "MAIN RETRIEVED CHUNK" in context
    assert "previous product text" in context
    assert "next product features" in context
    assert audit[0]["previous_chunk_ids"] == ["docA-1"]
    assert audit[0]["next_chunk_ids"] == ["docA-3"]
