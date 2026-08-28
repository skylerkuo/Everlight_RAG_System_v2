from rag_app.qa.filters import contains_exact_name
from rag_app.qa.query_tools import build_search_query, keywords_materially_changed


def test_exact_product_boundary():
    assert contains_exact_name("EL827 is a dual channel device", "EL827")
    assert not contains_exact_name("EL8270 is a different string", "EL827")


def test_keyword_change_ignores_order_and_case():
    assert not keywords_materially_changed(
        ["SMD", "四通道"],
        ["四通道", "smd"],
    )
    assert keywords_materially_changed(
        ["SMD", "四通道"],
        ["SMD", "four isolated channels"],
    )


def test_search_query_always_preserves_original_question():
    question = "如果要求 5000 Vrms 又是雙通道 Photo Transistor，哪個系列符合？"
    query = build_search_query(
        question,
        ["5000 Vrms", "雙通道", "Photo Transistor"],
        [],
    )
    assert query.startswith(question)
    assert "Search keywords:" in query


def test_exact_product_filter_keeps_same_document_candidates():
    from rag_app.qa.filters import apply_exact_product_filter
    from rag_app.retrieval.bge_m3_index import SearchResult

    # 同一產品頁被切成不同 section/chunks：只有第一塊出現產品型號，
    # 真正規格在第二塊；兩塊都應保留給 reranker。
    same_doc_name = SearchResult(
        rank=1,
        score=0.9,
        chunk={
            "document_id": "doc_el31x0",
            "content": "EL31X0 series is an 8-pin DIP gate driver.",
        },
    )
    same_doc_spec = SearchResult(
        rank=2,
        score=0.8,
        chunk={
            "document_id": "doc_el31x0",
            "content": "Guaranteed performance from -40°C to 110°C; isolation 5000Vrms.",
        },
    )
    other_doc = SearchResult(
        rank=3,
        score=0.7,
        chunk={
            "document_id": "doc_other",
            "content": "Other gate driver information.",
        },
    )

    filtered, applied = apply_exact_product_filter(
        [same_doc_name, same_doc_spec, other_doc],
        ["EL31X0"],
    )

    assert applied is True
    assert filtered == [same_doc_name, same_doc_spec]


def test_exact_product_filter_only_uses_original_candidate_pool():
    from rag_app.qa.filters import apply_exact_product_filter
    from rag_app.retrieval.bge_m3_index import SearchResult

    # Filter 只能重整傳入的 BGE candidates，不會額外去 index 擴張資料。
    candidate = SearchResult(
        rank=1,
        score=0.9,
        chunk={
            "document_id": "doc_elw3120",
            "content": "ELW3120 is an 8-pin wide body gate driver.",
        },
    )

    filtered, applied = apply_exact_product_filter([candidate], ["ELW3120"])

    assert applied is True
    assert filtered == [candidate]
