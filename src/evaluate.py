"""評価データ（質問 + 正解ページ）で検索精度を測定し、結果を JSON に保存する.

usage: python -m src.evaluate [--dataset FILE] [--top-k N] [--min-score F]
                              [--index-dir DIR] [--book NAME] [--output FILE] [--verbose]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from .common import (
    DEFAULT_INDEX_DIR,
    DEFAULT_TOP_K,
    PROJECT_ROOT,
    BookIndex,
    BookRagError,
    Embedder,
    load_all_indexes,
    setup_logging,
)
from .search import SearchResult, search

DEFAULT_DATASET = PROJECT_ROOT / "datasets" / "evaluation.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "evaluation.json"


@dataclass
class EvalCase:
    question: str
    expected_pages: list[int]
    note: str = ""


@dataclass
class CaseResult:
    question: str
    expected_pages: list[int]
    retrieved_pages: list[int]
    top_scores: list[float]
    top1_hit: bool
    top3_hit: bool
    search_time_sec: float
    note: str = ""


@dataclass
class EvalSummary:
    total: int
    top1_correct: int
    top1_accuracy: float
    top3_correct: int
    top3_accuracy: float
    avg_search_time_sec: float


def load_dataset(path: Path) -> list[EvalCase]:
    if not path.exists():
        raise BookRagError(f"評価データが見つかりません: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise BookRagError(f"評価データのJSONが不正です: {path} ({e})") from e
    if not isinstance(raw, list) or not raw:
        raise BookRagError(f"評価データは1件以上の配列である必要があります: {path}")
    cases: list[EvalCase] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict) or "question" not in item or "expected_pages" not in item:
            raise BookRagError(f"評価データ {i} 件目に question / expected_pages がありません")
        pages = item["expected_pages"]
        if not isinstance(pages, list) or not all(isinstance(p, int) for p in pages) or not pages:
            raise BookRagError(f"評価データ {i} 件目の expected_pages は整数の配列である必要があります")
        cases.append(EvalCase(question=str(item["question"]), expected_pages=pages, note=str(item.get("note", ""))))
    return cases


def judge(case: EvalCase, results: list[SearchResult], elapsed: float) -> CaseResult:
    pages = [r.page for r in results]
    expected = set(case.expected_pages)
    return CaseResult(
        question=case.question,
        expected_pages=case.expected_pages,
        retrieved_pages=pages,
        top_scores=[r.score for r in results],
        top1_hit=bool(pages) and pages[0] in expected,
        top3_hit=any(p in expected for p in pages[:3]),
        search_time_sec=round(elapsed, 4),
        note=case.note,
    )


def summarize(results: list[CaseResult]) -> EvalSummary:
    n = len(results)
    top1 = sum(r.top1_hit for r in results)
    top3 = sum(r.top3_hit for r in results)
    return EvalSummary(
        total=n,
        top1_correct=top1,
        top1_accuracy=round(top1 / n, 4) if n else 0.0,
        top3_correct=top3,
        top3_accuracy=round(top3 / n, 4) if n else 0.0,
        avg_search_time_sec=round(sum(r.search_time_sec for r in results) / n, 4) if n else 0.0,
    )


def evaluate(
    cases: list[EvalCase],
    embedder: Embedder,
    indexes: list[BookIndex],
    top_k: int = DEFAULT_TOP_K,
    min_score: float = 0.0,
) -> list[CaseResult]:
    if top_k < 3:
        raise BookRagError("--top-k は Top3 正解率を算出するため 3 以上を指定してください")
    results: list[CaseResult] = []
    for case in cases:
        start = time.perf_counter()
        hits = search(case.question, embedder, indexes, top_k, min_score)
        results.append(judge(case, hits, time.perf_counter() - start))
    return results


def build_report(
    case_results: list[CaseResult],
    indexes: list[BookIndex],
    top_k: int,
    min_score: float,
    dataset_path: Path,
) -> dict:
    metas = [idx.meta for idx in indexes]
    return {
        "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": str(dataset_path),
        "model": metas[0].model,
        "chunk_size": metas[0].chunk_size,
        "chunk_overlap": metas[0].chunk_overlap,
        "top_k": top_k,
        "min_score": min_score,
        "books": [
            {"book": m.book, "chunk_count": m.chunk_count, "processed_pages": m.processed_pages}
            for m in metas
        ],
        "summary": asdict(summarize(case_results)),
        "results": [asdict(r) for r in case_results],
        "failure_analysis": "",  # 失敗した質問の原因考察を手書きで追記する欄
    }


def format_report(report: dict) -> str:
    s = report["summary"]
    lines = [
        "=== 評価結果 ===",
        f"モデル        : {report['model']}",
        f"chunk_size    : {report['chunk_size']} / overlap: {report['chunk_overlap']} / top_k: {report['top_k']}",
        f"質問数        : {s['total']}",
        f"Top1 正解     : {s['top1_correct']}/{s['total']} ({s['top1_accuracy']:.1%})",
        f"Top3 正解     : {s['top3_correct']}/{s['total']} ({s['top3_accuracy']:.1%})",
        f"平均検索時間  : {s['avg_search_time_sec']:.3f} 秒/質問",
    ]
    failed = [r for r in report["results"] if not r["top3_hit"]]
    if failed:
        lines.append("")
        lines.append("--- Top3 で正解ページを取得できなかった質問 ---")
        for r in failed:
            lines.append(f"Q: {r['question']}")
            lines.append(f"   期待ページ: {r['expected_pages']} / 取得ページ: {r['retrieved_pages']} / score: {r['top_scores']}")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="評価データで検索精度を測定する")
    p.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="評価データJSON")
    p.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    p.add_argument("--min-score", type=float, default=0.0, help="評価時の最低類似度（既定: フィルタなし）")
    p.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    p.add_argument("--book", help="評価対象を特定の書籍（PDFファイル名）に絞る")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="結果JSONの保存先")
    p.add_argument("--verbose", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.verbose)
    try:
        cases = load_dataset(args.dataset)
        indexes = load_all_indexes(args.index_dir, args.book)
        embedder = Embedder(indexes[0].meta.model)
        case_results = evaluate(cases, embedder, indexes, args.top_k, args.min_score)
    except BookRagError as e:
        print(f"エラー: {e}", file=sys.stderr)
        return 1

    report = build_report(case_results, indexes, args.top_k, args.min_score, args.dataset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(format_report(report))
    print(f"\n結果を保存しました: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
