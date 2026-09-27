"""PDF を読み込み、ページ単位で本文を抽出 → チャンク化 → ベクトル化 → 保存する.

usage: python -m src.ingest <pdf_path> [--chunk-size N] [--chunk-overlap N]
                            [--model NAME] [--index-dir DIR]
                            [--ocr auto|off|force] [--ocr-lang auto|jpn|jpn_vert] [--verbose]

本文が取れないページ（スキャン画像など）は --ocr auto（既定）で Tesseract により OCR する。
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from .common import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_INDEX_DIR,
    DEFAULT_MODEL,
    BookIndex,
    BookRagError,
    Chunk,
    Embedder,
    IndexMeta,
    logger,
    save_index,
    setup_logging,
)
from .ocr import DEFAULT_OCR_LANG, DEFAULT_OCR_MODE, OCR_LANGS, OCR_MODES, PageOcr

# チャンク分割時に優先する境界（後ろにあるものほど弱い境界）
_BOUNDARY_PATTERN = re.compile(r"[。！？\n]|[.!?](?=\s)")


@dataclass
class PageText:
    page: int  # 1始まりのPDFページ番号
    text: str


@dataclass
class ExtractionResult:
    book: str
    total_pages: int
    pages: list[PageText]
    skipped_pages: list[int]
    ocr_pages: list[int] | None = None


def _normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t\u3000]+", " ", text)
    # 日本語は行末で単語が折り返されるため、全角文字間の改行は連結する（段落境界の空行は残す）
    text = re.sub(r"(?<=[^\x00-\x7F])\n(?=[^\x00-\x7F\n])", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _has_image(page) -> bool:
    try:
        resources = page.get("/Resources") or {}
        xobjects = resources.get("/XObject")
        if xobjects is None:
            return False
        xobjects = xobjects.get_object()
        return any(xobjects[name].get_object().get("/Subtype") == "/Image" for name in xobjects)
    except Exception:  # noqa: BLE001
        return False


def extract_pages(
    pdf_path: Path,
    ocr_mode: str = DEFAULT_OCR_MODE,
    ocr_lang: str = DEFAULT_OCR_LANG,
) -> ExtractionResult:
    """PDF からページごとの本文を抽出する. 空ページは除外し警告を記録する.

    ocr_mode: auto=本文が取れないページのみ OCR / off=OCR しない / force=全ページ OCR
    """
    if ocr_mode not in OCR_MODES:
        raise BookRagError(f"--ocr は {', '.join(OCR_MODES)} のいずれかを指定してください")
    if not pdf_path.exists():
        raise BookRagError(f"PDFファイルが見つかりません: {pdf_path}")
    if not pdf_path.is_file():
        raise BookRagError(f"ファイルではありません: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise BookRagError(f"PDFファイル（.pdf）を指定してください: {pdf_path}")

    try:
        reader = PdfReader(str(pdf_path))
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception as e:  # noqa: BLE001
                raise BookRagError(f"暗号化されたPDFは読み込めません: {pdf_path}") from e
        total_pages = len(reader.pages)
    except PdfReadError as e:
        raise BookRagError(f"PDFとして読み込めません（破損または非対応形式）: {pdf_path} ({e})") from e

    ocr: PageOcr | None = None
    ocr_unavailable = False
    if ocr_mode == "force":
        ocr = PageOcr(pdf_path, ocr_lang)

    pages: list[PageText] = []
    skipped: list[int] = []
    ocr_pages: list[int] = []
    for i, page in enumerate(reader.pages, start=1):
        text = ""
        if ocr_mode != "force":
            try:
                text = _normalize_text(page.extract_text() or "")
            except Exception as e:  # noqa: BLE001
                logger.warning("ページ %d のテキスト抽出に失敗しました: %s", i, e)
        # auto: 文字が無く画像を含むページ（スキャン画像）のみ OCR する。白紙ページは対象外
        needs_ocr = ocr_mode == "force" or (ocr_mode == "auto" and not text and _has_image(page))
        if needs_ocr and not ocr_unavailable:
            if ocr is None:
                logger.info("本文が取れないページがあるため OCR を実行します（時間がかかります）")
                try:
                    ocr = PageOcr(pdf_path, ocr_lang)
                except BookRagError as e:
                    if ocr_mode == "force":
                        raise
                    logger.warning("OCR をスキップします: %s", e)
                    ocr_unavailable = True
            if ocr is not None:
                logger.debug("ページ %d を OCR 中", i)
                text = _normalize_text(ocr.ocr_page(i))
                if text:
                    ocr_pages.append(i)
        if not text:
            logger.warning("ページ %d は本文を抽出できなかったため除外します", i)
            skipped.append(i)
            continue
        pages.append(PageText(page=i, text=text))

    if not pages:
        hint = (
            "画像のみのPDFの場合は --ocr auto（既定）で OCR を試してください。"
            if ocr_mode == "off"
            else "OCR でも文字を認識できませんでした。画像の向き・解像度を確認してください。"
        )
        raise BookRagError(f"本文を1ページも抽出できませんでした: {pdf_path}\n{hint}")
    if ocr_pages:
        logger.info("OCR で本文を取得したページ: %d ページ", len(ocr_pages))
    return ExtractionResult(
        book=pdf_path.name,
        total_pages=total_pages,
        pages=pages,
        skipped_pages=skipped,
        ocr_pages=ocr_pages,
    )


def split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """文字数ベースでテキストを分割する. 可能なら文末（。！？改行）で区切る."""
    if chunk_size <= 0:
        raise BookRagError("--chunk-size は 1 以上を指定してください")
    if overlap < 0 or overlap >= chunk_size:
        raise BookRagError("--chunk-overlap は 0 以上かつ chunk-size 未満を指定してください")

    text = text.strip()
    if len(text) <= chunk_size:
        return [text] if text else []

    chunks: list[str] = []
    start = 0
    min_cut = max(1, chunk_size // 2)  # これより短くなる位置での境界分割はしない
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            window = text[start:end]
            cut = -1
            for m in _BOUNDARY_PATTERN.finditer(window):
                if m.end() >= min_cut:
                    cut = m.end()
            if cut > 0:
                end = start + cut
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def build_chunks(result: ExtractionResult, chunk_size: int, overlap: int) -> list[Chunk]:
    stem = Path(result.book).stem
    chunks: list[Chunk] = []
    for page in result.pages:
        for n, piece in enumerate(split_text(page.text, chunk_size, overlap), start=1):
            chunks.append(
                Chunk(
                    book=result.book,
                    page=page.page,
                    chunk_id=f"{stem}-p{page.page}-c{n:02d}",
                    text=piece,
                )
            )
    return chunks


def ingest(
    pdf_path: Path,
    embedder: Embedder,
    index_dir: Path = DEFAULT_INDEX_DIR,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    extraction: ExtractionResult | None = None,
) -> BookIndex:
    result = extraction or extract_pages(pdf_path)
    logger.info(
        "抽出完了: %s 全%dページ / 処理%dページ / 除外%dページ",
        result.book,
        result.total_pages,
        len(result.pages),
        len(result.skipped_pages),
    )
    chunks = build_chunks(result, chunk_size, chunk_overlap)
    logger.info("チャンク生成: %d件 (chunk_size=%d, overlap=%d)", len(chunks), chunk_size, chunk_overlap)

    embeddings = embedder.embed_passages([c.text for c in chunks])
    meta = IndexMeta(
        book=result.book,
        source_path=str(pdf_path.resolve()),
        model=embedder.model_name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        total_pages=result.total_pages,
        processed_pages=len(result.pages),
        skipped_pages=len(result.skipped_pages),
        skipped_page_numbers=result.skipped_pages,
        chunk_count=len(chunks),
        embedding_dim=int(np.asarray(embeddings).shape[1]),
    )
    index = BookIndex(meta=meta, chunks=chunks, embeddings=np.asarray(embeddings))
    saved_to = save_index(index, index_dir)
    logger.info("インデックス保存先: %s", saved_to)
    return index


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="PDFを取り込み検索用インデックスを作成する")
    p.add_argument("pdf_path", type=Path, help="取り込むPDFファイルのパス")
    p.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, help="チャンク文字数")
    p.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP, help="チャンク間の重複文字数")
    p.add_argument("--model", default=DEFAULT_MODEL, help="Embeddingモデル名")
    p.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR, help="インデックス保存先")
    p.add_argument(
        "--ocr",
        choices=OCR_MODES,
        default=DEFAULT_OCR_MODE,
        help="auto: 本文が取れないページのみOCR（既定） / off: OCRしない / force: 全ページOCR",
    )
    p.add_argument(
        "--ocr-lang",
        choices=OCR_LANGS,
        default=DEFAULT_OCR_LANG,
        help="OCR言語. auto: 最初のOCRページで横書き(jpn)/縦書き(jpn_vert)を自動判定（既定）",
    )
    p.add_argument("--verbose", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.verbose)
    try:
        # モデルロード前に入力を検証し、不備があれば即座にエラーを返す
        split_text("x" * (args.chunk_size + 1), args.chunk_size, args.chunk_overlap)
        extraction = extract_pages(args.pdf_path, args.ocr, args.ocr_lang)
        embedder = Embedder(args.model)
        index = ingest(
            args.pdf_path, embedder, args.index_dir, args.chunk_size, args.chunk_overlap, extraction
        )
    except BookRagError as e:
        print(f"エラー: {e}", file=sys.stderr)
        return 1

    m = index.meta
    print("=== 取り込み結果 ===")
    print(f"書籍名      : {m.book}")
    print(f"全ページ数  : {m.total_pages}")
    print(f"処理ページ数: {m.processed_pages}")
    print(f"除外ページ数: {m.skipped_pages}" + (f" {m.skipped_page_numbers}" if m.skipped_pages else ""))
    print(f"OCRページ数  : {len(extraction.ocr_pages or [])}")
    print(f"生成チャンク: {m.chunk_count}")
    print(f"モデル      : {m.model}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
