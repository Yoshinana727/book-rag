"""質問文をベクトル化し、保存済みインデックスから類似チャンクを検索する.

usage: python -m src.search "<質問文>" [--top-k N] [--min-score F]
                            [--index-dir DIR] [--book NAME] [--json] [--verbose]
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .common import (
    DEFAULT_INDEX_DIR,
    DEFAULT_MIN_SCORE,
    DEFAULT_TOP_K,
    BookIndex,
    BookRagError,
    Embedder,
    load_all_indexes,
    logger,
    setup_logging,
)


@dataclass
class SearchResult:
    rank: int
    book: str
    page: int
    chunk_id: str
    score: float
    text: str


def search_indexes(
    query_vec: np.ndarray,
    indexes: list[BookIndex],
    top_k: int = DEFAULT_TOP_K,
    min_score: float = DEFAULT_MIN_SCORE,
) -> list[SearchResult]:
    """正規化済みクエリベクトルとインデックス群からコサイン類似度上位 top_k 件を返す."""
    if top_k <= 0:
        raise BookRagError("--top-k は 1 以上を指定してください")
    candidates: list[tuple[float, BookIndex, int]] = []
    for index in indexes:
        scores = index.embeddings @ query_vec  # 両者正規化済みなので内積 = コサイン類似度
        k = min(top_k, len(scores))
        top_idx = np.argpartition(-scores, k - 1)[:k] if k < len(scores) else np.arange(len(scores))
        candidates.extend((float(scores[i]), index, int(i)) for i in top_idx)

    candidates.sort(key=lambda c: c[0], reverse=True)
    results: list[SearchResult] = []
    for score, index, i in candidates[:top_k]:
        if score < min_score:
            continue
        c = index.chunks[i]
        results.append(
            SearchResult(
                rank=len(results) + 1,
                book=c.book,
                page=c.page,
                chunk_id=c.chunk_id,
                score=round(score, 4),
                text=c.text,
            )
        )
    return results


def search(
    query: str,
    embedder: Embedder,
    indexes: list[BookIndex],
    top_k: int = DEFAULT_TOP_K,
    min_score: float = DEFAULT_MIN_SCORE,
) -> list[SearchResult]:
    if not query or not query.strip():
        raise BookRagError("質問文が空です")
    return search_indexes(embedder.embed_query(query.strip()), indexes, top_k, min_score)


def format_results(results: list[SearchResult], min_score: float) -> str:
    if not results:
        return f"該当する本文が見つかりませんでした（類似度が基準値 {min_score:.2f} を下回りました）"
    blocks = []
    for r in results:
        blocks.append(f"[{r.rank}] {r.book} / {r.page}ページ / score: {r.score:.2f}\n{r.text}")
    return "\n\n".join(blocks)


def check_model_consistency(indexes: list[BookIndex], model_name: str) -> None:
    for idx in indexes:
        if idx.meta.model != model_name:
            logger.warning(
                "インデックス '%s' は %s で作成されていますが、検索には %s を使用しています。"
                "精度が低下する可能性があります。",
                idx.meta.book,
                idx.meta.model,
                model_name,
            )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="質問文に近い本文をページ番号付きで検索する")
    p.add_argument("query", help="質問文")
    p.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="表示件数")
    p.add_argument("--min-score", type=float, default=DEFAULT_MIN_SCORE, help="表示する最低類似度")
    p.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    p.add_argument("--book", help="検索対象を特定の書籍（PDFファイル名）に絞る")
    p.add_argument("--model", help="Embeddingモデル名（省略時はインデックス作成時のモデル）")
    p.add_argument("--json", action="store_true", help="結果をJSONで出力する")
    p.add_argument("--verbose", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.verbose)
    try:
        indexes = load_all_indexes(args.index_dir, args.book)
        model_name = args.model or indexes[0].meta.model
        check_model_consistency(indexes, model_name)
        embedder = Embedder(model_name)
        results = search(args.query, embedder, indexes, args.top_k, args.min_score)
    except BookRagError as e:
        print(f"エラー: {e}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps([asdict(r) for r in results], ensure_ascii=False, indent=2))
    else:
        print(format_results(results, args.min_score))
    return 0


if __name__ == "__main__":
    sys.exit(main())
