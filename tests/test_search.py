from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.common import BookIndex, BookRagError, Chunk, IndexMeta, load_all_indexes
from src.ingest import ingest
from src.search import format_results, search, search_indexes


def _meta(book: str, n: int) -> IndexMeta:
    return IndexMeta(
        book=book,
        source_path=book,
        model="fake-embedder",
        chunk_size=100,
        chunk_overlap=0,
        created_at="2026-01-01T00:00:00+00:00",
        total_pages=n,
        processed_pages=n,
        skipped_pages=0,
        skipped_page_numbers=[],
        chunk_count=n,
        embedding_dim=2,
    )


def _index(book: str, vectors: list[list[float]]) -> BookIndex:
    emb = np.asarray(vectors, dtype=np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    chunks = [Chunk(book=book, page=i + 1, chunk_id=f"{book}-p{i + 1}-c01", text=f"text{i + 1}") for i in range(len(vectors))]
    return BookIndex(meta=_meta(book, len(vectors)), chunks=chunks, embeddings=emb)


def test_search_indexes_orders_by_cosine_similarity():
    idx = _index("a.pdf", [[1, 0], [0, 1], [1, 1], [-1, 0]])
    q = np.asarray([1, 0.2], dtype=np.float32)
    q /= np.linalg.norm(q)
    results = search_indexes(q, [idx], top_k=3, min_score=-1)
    assert [r.page for r in results] == [1, 3, 2]
    assert [r.rank for r in results] == [1, 2, 3]
    assert results[0].score >= results[1].score >= results[2].score


def test_search_indexes_respects_top_k_and_min_score():
    idx = _index("a.pdf", [[1, 0], [0, 1], [-1, 0]])
    q = np.asarray([1, 0], dtype=np.float32)
    assert len(search_indexes(q, [idx], top_k=2, min_score=-1)) == 2
    # score: 1.0, 0.0, -1.0 → 0.5 以上は1件
    results = search_indexes(q, [idx], top_k=3, min_score=0.5)
    assert [r.page for r in results] == [1]
    assert search_indexes(q, [idx], top_k=3, min_score=1.5) == []


def test_search_indexes_across_multiple_books():
    a = _index("a.pdf", [[1, 0]])
    b = _index("b.pdf", [[0.9, 0.1], [0, 1]])
    q = np.asarray([1, 0], dtype=np.float32)
    results = search_indexes(q, [a, b], top_k=2, min_score=-1)
    assert [(r.book, r.page) for r in results] == [("a.pdf", 1), ("b.pdf", 1)]


def test_search_indexes_invalid_top_k():
    idx = _index("a.pdf", [[1, 0]])
    with pytest.raises(BookRagError):
        search_indexes(np.asarray([1, 0], dtype=np.float32), [idx], top_k=0)


def test_search_empty_query(fake_embedder):
    idx = _index("a.pdf", [[1, 0]])
    with pytest.raises(BookRagError, match="空"):
        search("   ", fake_embedder, [idx])


def test_format_results_matches_spec_layout():
    idx = _index("sample.pdf", [[1, 0]])
    idx.chunks[0].text = "該当する本文がここに表示されます。"
    idx.chunks[0].page = 10
    q = np.asarray([1, 0], dtype=np.float32)
    out = format_results(search_indexes(q, [idx], top_k=1, min_score=0), 0.3)
    assert out.startswith("[1] sample.pdf / 10ページ / score: 1.00\n該当する本文がここに表示されます。")


def test_format_results_no_hit_message():
    out = format_results([], 0.3)
    assert "見つかりませんでした" in out and "0.30" in out


def test_end_to_end_ingest_then_search(sample_pdf: Path, tmp_path: Path, fake_embedder):
    index_dir = tmp_path / "index"
    ingest(sample_pdf, fake_embedder, index_dir, chunk_size=200, chunk_overlap=0)
    indexes = load_all_indexes(index_dir)
    assert len(indexes) == 1

    target = next(c for c in indexes[0].chunks if "雪国" in c.text)
    results = search(f"SIMILAR_TO:{target.text}", fake_embedder, indexes, top_k=3, min_score=0.0)
    assert results[0].page == target.page
    assert results[0].book == "sample.pdf"
    assert "雪国" in results[0].text


def test_load_all_indexes_errors(tmp_path: Path):
    with pytest.raises(BookRagError, match="ingest"):
        load_all_indexes(tmp_path / "missing")
    (tmp_path / "empty").mkdir()
    with pytest.raises(BookRagError, match="1件もありません"):
        load_all_indexes(tmp_path / "empty")


def test_load_all_indexes_filter_by_book(sample_pdf: Path, tmp_path: Path, fake_embedder):
    index_dir = tmp_path / "index"
    ingest(sample_pdf, fake_embedder, index_dir)
    assert load_all_indexes(index_dir, "sample.pdf")[0].meta.book == "sample.pdf"
    with pytest.raises(BookRagError, match="見つかりません"):
        load_all_indexes(index_dir, "other.pdf")
