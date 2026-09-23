"""ingest / search / evaluate で共有する定数・ユーティリティ."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INDEX_DIR = PROJECT_ROOT / "data" / "index"
DEFAULT_MODEL = "intfloat/multilingual-e5-small"
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 100
DEFAULT_TOP_K = 3
DEFAULT_MIN_SCORE = 0.75

CHUNKS_FILE = "chunks.json"
EMBEDDINGS_FILE = "embeddings.npy"
META_FILE = "meta.json"

logger = logging.getLogger("book_rag")


def setup_logging(verbose: bool = False) -> None:
    # ライブラリ（httpx / sentence_transformers 等）の INFO ログは抑制し、自前のログのみ出す
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)


class BookRagError(Exception):
    """ユーザーに原因を提示して終了するためのエラー."""


@dataclass
class Chunk:
    book: str
    page: int
    chunk_id: str
    text: str


@dataclass
class IndexMeta:
    book: str
    source_path: str
    model: str
    chunk_size: int
    chunk_overlap: int
    created_at: str
    total_pages: int
    processed_pages: int
    skipped_pages: int
    skipped_page_numbers: list[int]
    chunk_count: int
    embedding_dim: int


@dataclass
class BookIndex:
    meta: IndexMeta
    chunks: list[Chunk]
    embeddings: np.ndarray  # shape: (len(chunks), dim), L2 正規化済み


class Embedder:
    """sentence-transformers のラッパー. e5 系の prefix 規約を隠蔽する."""

    def __init__(self, model_name: str = DEFAULT_MODEL):
        # 重い import はモデルを実際に使うときだけ行う
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        logger.info("Embeddingモデルをロード中: %s", model_name)
        self.model = SentenceTransformer(model_name)

    def _encode(self, texts: list[str], prefix: str) -> np.ndarray:
        vectors = self.model.encode(
            [prefix + t for t in texts],
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 32,
            convert_to_numpy=True,
        )
        return np.asarray(vectors, dtype=np.float32)

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, "passage: ")

    def embed_query(self, text: str) -> np.ndarray:
        return self._encode([text], "query: ")[0]


def save_index(index: BookIndex, index_dir: Path) -> Path:
    target = index_dir / Path(index.meta.book).stem
    target.mkdir(parents=True, exist_ok=True)
    (target / CHUNKS_FILE).write_text(
        json.dumps([asdict(c) for c in index.chunks], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    np.save(target / EMBEDDINGS_FILE, index.embeddings.astype(np.float32))
    (target / META_FILE).write_text(
        json.dumps(asdict(index.meta), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return target


def load_index(book_dir: Path) -> BookIndex:
    try:
        meta = IndexMeta(**json.loads((book_dir / META_FILE).read_text(encoding="utf-8")))
        chunks = [
            Chunk(**c) for c in json.loads((book_dir / CHUNKS_FILE).read_text(encoding="utf-8"))
        ]
        embeddings = np.load(book_dir / EMBEDDINGS_FILE)
    except FileNotFoundError as e:
        raise BookRagError(f"インデックスが壊れています（ファイル不足）: {e.filename}") from e
    if len(chunks) != embeddings.shape[0]:
        raise BookRagError(
            f"インデックスが壊れています: chunks={len(chunks)} embeddings={embeddings.shape[0]} ({book_dir})"
        )
    return BookIndex(meta=meta, chunks=chunks, embeddings=embeddings)


def load_all_indexes(index_dir: Path, book: str | None = None) -> list[BookIndex]:
    if not index_dir.exists():
        raise BookRagError(
            f"インデックスディレクトリが存在しません: {index_dir}\n"
            "先に `python -m src.ingest <pdf_path>` を実行してください。"
        )
    book_dirs = sorted(d for d in index_dir.iterdir() if (d / META_FILE).exists())
    if book is not None:
        stem = Path(book).stem
        book_dirs = [d for d in book_dirs if d.name == stem]
        if not book_dirs:
            raise BookRagError(f"書籍 '{book}' のインデックスが見つかりません: {index_dir}")
    if not book_dirs:
        raise BookRagError(
            f"インデックスが1件もありません: {index_dir}\n"
            "先に `python -m src.ingest <pdf_path>` を実行してください。"
        )
    return [load_index(d) for d in book_dirs]
