from __future__ import annotations

from pathlib import Path

import pytest

import src.ingest as ingest_mod
import src.ocr as ocr_mod
from src.common import BookRagError
from src.ingest import extract_pages
from src.ocr import clean_ocr_text, create_page_ocr, ja_char_count, order_vision_lines


def test_clean_ocr_text_removes_spaces_between_japanese_chars():
    raw = "これ は 、 コ ン サ ル タ ン ト です 。 a と b があります 。 \n次 の 行  \n"
    assert clean_ocr_text(raw) == "これは、コンサルタントです。 a と b があります。\n次の行"


def test_ja_char_count_counts_only_japanese():
    assert ja_char_count("abc 123 あい漢字カナ") == 6


def test_image_pdf_with_ocr_off_raises(image_pdf: Path):
    with pytest.raises(BookRagError, match="--ocr auto"):
        extract_pages(image_pdf, ocr_mode="off")


def test_image_pdf_auto_without_tesseract_skips_with_warning(image_pdf: Path, monkeypatch, caplog):
    monkeypatch.setattr("src.ocr.shutil.which", lambda _: None)
    with caplog.at_level("WARNING", logger="book_rag"):
        with pytest.raises(BookRagError, match="OCR でも文字を認識できませんでした|1ページも"):
            extract_pages(image_pdf, ocr_mode="auto")
    assert any("OCR をスキップします" in r.message for r in caplog.records)


def test_force_without_tesseract_raises_install_hint(image_pdf: Path, monkeypatch):
    monkeypatch.setattr("src.ocr.shutil.which", lambda _: None)
    with pytest.raises(BookRagError, match="brew install tesseract"):
        extract_pages(image_pdf, ocr_mode="force")


class FakePageOcr:
    calls: list[int] = []

    def __init__(self, pdf_path: Path, lang: str = "auto", engine: str = "auto"):
        self.lang = lang
        self.engine = engine

    def ocr_page(self, page_no: int) -> str:
        FakePageOcr.calls.append(page_no)
        return f"OCRされた本文 {page_no}ページ目。"


def test_auto_mode_ocrs_only_image_pages(image_pdf: Path, sample_pdf: Path, monkeypatch):
    monkeypatch.setattr(ingest_mod, "create_page_ocr", FakePageOcr)

    FakePageOcr.calls = []
    result = extract_pages(image_pdf, ocr_mode="auto")
    assert FakePageOcr.calls == [1, 2]
    assert [p.page for p in result.pages] == [1, 2]
    assert result.ocr_pages == [1, 2]
    assert result.pages[0].text == "OCRされた本文 1ページ目。"

    # テキストPDFの白紙ページ（画像なし）は OCR 対象にならない
    FakePageOcr.calls = []
    result = extract_pages(sample_pdf, ocr_mode="auto")
    assert FakePageOcr.calls == []
    assert result.skipped_pages == [2]
    assert result.ocr_pages == []


def test_force_mode_ocrs_every_page(sample_pdf: Path, monkeypatch):
    monkeypatch.setattr(ingest_mod, "create_page_ocr", FakePageOcr)
    FakePageOcr.calls = []
    result = extract_pages(sample_pdf, ocr_mode="force")
    assert FakePageOcr.calls == [1, 2, 3, 4]
    assert all(p.text.startswith("OCRされた本文") for p in result.pages)


def test_invalid_ocr_mode(sample_pdf: Path):
    with pytest.raises(BookRagError, match="--ocr は"):
        extract_pages(sample_pdf, ocr_mode="maybe")


def test_invalid_ocr_engine(image_pdf: Path):
    with pytest.raises(BookRagError, match="--ocr-engine"):
        create_page_ocr(image_pdf, engine="gpt")


def test_vision_engine_outside_macos_raises_hint(image_pdf: Path, monkeypatch):
    monkeypatch.setattr(ocr_mod.sys, "platform", "linux")
    with pytest.raises(BookRagError, match="macOS 専用"):
        create_page_ocr(image_pdf, engine="vision")


def test_auto_engine_on_macos_without_ocrmac_falls_back_to_tesseract(image_pdf: Path, monkeypatch, caplog):
    monkeypatch.setattr(ocr_mod.sys, "platform", "darwin")
    monkeypatch.setattr(ocr_mod, "vision_available", lambda: False)
    monkeypatch.setattr(ocr_mod, "TesseractPageOcr", lambda pdf_path, lang: ("tesseract", lang))
    with caplog.at_level("WARNING", logger="book_rag"):
        assert create_page_ocr(image_pdf, lang="jpn") == ("tesseract", "jpn")
    assert any("pip install ocrmac" in r.message for r in caplog.records)


def test_auto_engine_prefers_vision_when_available(image_pdf: Path, monkeypatch):
    monkeypatch.setattr(ocr_mod, "vision_available", lambda: True)
    monkeypatch.setattr(ocr_mod, "VisionPageOcr", lambda pdf_path, lang: ("vision", lang))
    assert create_page_ocr(image_pdf) == ("vision", "auto")


def test_order_vision_lines_horizontal_top_to_bottom():
    # Vision の座標は原点左下。y が大きいほど上
    lines = [
        ("二行目", 1.0, [0.1, 0.70, 0.8, 0.05]),
        ("一行目", 1.0, [0.1, 0.80, 0.8, 0.05]),
        ("三行目", 1.0, [0.1, 0.60, 0.8, 0.05]),
    ]
    assert order_vision_lines(lines) == ["一行目", "二行目", "三行目"]


def test_order_vision_lines_vertical_right_to_left():
    lines = [
        ("二行目", 1.0, [0.50, 0.1, 0.05, 0.8]),
        ("一行目", 1.0, [0.80, 0.1, 0.05, 0.8]),
        ("三行目", 1.0, [0.20, 0.1, 0.05, 0.8]),
    ]
    assert order_vision_lines(lines) == ["一行目", "二行目", "三行目"]
    assert order_vision_lines([]) == []
