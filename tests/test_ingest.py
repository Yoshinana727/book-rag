from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.common import BookRagError, load_index
from src.ingest import build_chunks, extract_pages, ingest, main, split_text


def test_extract_pages_returns_page_numbers_and_text(sample_pdf: Path):
    result = extract_pages(sample_pdf)
    assert result.book == "sample.pdf"
    assert result.total_pages == 4
    assert [p.page for p in result.pages] == [1, 3, 4]
    assert "吾輩は猫である" in result.pages[0].text
    assert "雪国" in result.pages[2].text


def test_extract_pages_skips_empty_pages_with_warning(sample_pdf: Path, caplog):
    with caplog.at_level("WARNING"):
        result = extract_pages(sample_pdf)
    assert result.skipped_pages == [2]
    assert "ページ 2" in caplog.text


def test_extract_pages_missing_file(tmp_path: Path):
    with pytest.raises(BookRagError, match="見つかりません"):
        extract_pages(tmp_path / "nope.pdf")


def test_extract_pages_non_pdf_extension(tmp_path: Path):
    f = tmp_path / "book.txt"
    f.write_text("hello")
    with pytest.raises(BookRagError, match=r"\.pdf"):
        extract_pages(f)


def test_extract_pages_corrupted_pdf(tmp_path: Path):
    f = tmp_path / "broken.pdf"
    f.write_bytes(b"this is not a pdf")
    with pytest.raises(BookRagError, match="読み込めません"):
        extract_pages(f)


def test_extract_pages_all_empty(tmp_path: Path):
    from tests.conftest import make_pdf

    pdf = make_pdf(tmp_path / "empty.pdf", [None, None])
    with pytest.raises(BookRagError, match="1ページも抽出できませんでした"):
        extract_pages(pdf)


def test_split_text_short_text_single_chunk():
    assert split_text("短い文。", 100, 10) == ["短い文。"]
    assert split_text("   ", 100, 10) == []


def test_split_text_respects_size_and_prefers_sentence_boundary():
    text = "あ" * 40 + "。" + "い" * 40 + "。" + "う" * 40 + "。"
    chunks = split_text(text, 60, 10)
    assert len(chunks) >= 2
    assert all(len(c) <= 60 for c in chunks)
    assert chunks[0].endswith("。")
    # 全文が網羅されている
    joined = "".join(chunks)
    assert "あ" * 40 in joined and "い" * 40 in joined and "う" * 40 in joined


def test_split_text_overlap():
    text = "x" * 250
    chunks = split_text(text, 100, 20)
    assert [len(c) for c in chunks] == [100, 100, 90]
    assert chunks[1][:20] == chunks[0][-20:]


def test_split_text_invalid_args():
    with pytest.raises(BookRagError):
        split_text("abc", 0, 0)
    with pytest.raises(BookRagError):
        split_text("abc", 10, 10)


def test_build_chunks_metadata(sample_pdf: Path):
    result = extract_pages(sample_pdf)
    chunks = build_chunks(result, chunk_size=20, overlap=5)
    assert chunks[0].book == "sample.pdf"
    assert chunks[0].page == 1
    assert chunks[0].chunk_id == "sample-p1-c01"
    assert chunks[1].chunk_id == "sample-p1-c02"
    pages = {c.page for c in chunks}
    assert pages == {1, 3, 4}
    assert all(c.text for c in chunks)


def test_ingest_saves_index(sample_pdf: Path, tmp_path: Path, fake_embedder):
    index_dir = tmp_path / "index"
    index = ingest(sample_pdf, fake_embedder, index_dir, chunk_size=50, chunk_overlap=10)

    saved = index_dir / "sample"
    assert (saved / "chunks.json").exists()
    assert (saved / "embeddings.npy").exists()
    meta = json.loads((saved / "meta.json").read_text(encoding="utf-8"))
    assert meta["book"] == "sample.pdf"
    assert meta["model"] == "fake-embedder"
    assert meta["chunk_size"] == 50 and meta["chunk_overlap"] == 10
    assert meta["total_pages"] == 4
    assert meta["processed_pages"] == 3
    assert meta["skipped_pages"] == 1 and meta["skipped_page_numbers"] == [2]
    assert meta["chunk_count"] == len(index.chunks)
    assert "created_at" in meta

    loaded = load_index(saved)
    assert len(loaded.chunks) == len(index.chunks)
    assert loaded.embeddings.shape == (len(index.chunks), fake_embedder.dim)
    np.testing.assert_allclose(np.linalg.norm(loaded.embeddings, axis=1), 1.0, atol=1e-5)


def test_main_missing_file_returns_error(tmp_path: Path, capsys):
    rc = main([str(tmp_path / "missing.pdf"), "--index-dir", str(tmp_path / "idx")])
    assert rc == 1
    assert "見つかりません" in capsys.readouterr().err
