from rag_app.config import Settings


def test_runtime_settings_are_consistent():
    settings = Settings.fixed()
    assert settings.candidate_k >= settings.top_k >= 1
    assert abs(
        settings.rrf_dense_weight + settings.rrf_sparse_weight - 1.0
    ) < 1e-6
    assert 1 <= settings.max_search_rounds <= 3
