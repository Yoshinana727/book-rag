from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.common import BookRagError, load_all_indexes
from src.evaluate import EvalCase, build_report, evaluate, judge, load_dataset, main, summarize
from src.ingest import ingest
from src.search import SearchResult


def _r(page: int, score: float = 0.5) -> SearchResult:
    return SearchResult(rank=0, book="b.pdf", page=page, chunk_id="x", score=score, text="t")


def test_judge_top1_and_top3():
    case = EvalCase(question="q", expected_pages=[10])
    hit1 = judge(case, [_r(10), _r(2), _r(3)], 0.1)
    assert hit1.top1_hit and hit1.top3_hit
    hit3 = judge(case, [_r(1), _r(2), _r(10)], 0.1)
    assert not hit3.top1_hit and hit3.top3_hit
    miss = judge(case, [_r(1), _r(2), _r(3), _r(10)], 0.1)
    assert not miss.top1_hit and not miss.top3_hit
    assert miss.retrieved_pages == [1, 2, 3, 10]
    none = judge(case, [], 0.1)
    assert not none.top1_hit and not none.top3_hit


def test_summarize():
    case = EvalCase(question="q", expected_pages=[1])
    results = [
        judge(case, [_r(1)], 0.2),
        judge(case, [_r(2), _r(1)], 0.4),
        judge(case, [_r(5)], 0.6),
    ]
    s = summarize(results)
    assert s.total == 3
    assert s.top1_correct == 1 and s.top1_accuracy == pytest.approx(0.3333, abs=1e-4)
    assert s.top3_correct == 2 and s.top3_accuracy == pytest.approx(0.6667, abs=1e-4)
    assert s.avg_search_time_sec == pytest.approx(0.4)


def test_load_dataset_validation(tmp_path: Path):
    with pytest.raises(BookRagError, match="見つかりません"):
        load_dataset(tmp_path / "none.json")

    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(BookRagError, match="不正"):
        load_dataset(bad)

    bad.write_text(json.dumps([{"question": "q"}]), encoding="utf-8")
    with pytest.raises(BookRagError, match="expected_pages"):
        load_dataset(bad)

    bad.write_text(json.dumps([{"question": "q", "expected_pages": ["1"]}]), encoding="utf-8")
    with pytest.raises(BookRagError, match="整数"):
        load_dataset(bad)

    ok = tmp_path / "ok.json"
    ok.write_text(json.dumps([{"question": "q", "expected_pages": [1, 2], "note": "n"}]), encoding="utf-8")
    cases = load_dataset(ok)
    assert cases[0].question == "q" and cases[0].expected_pages == [1, 2] and cases[0].note == "n"


def test_evaluate_and_report_end_to_end(sample_pdf: Path, tmp_path: Path, fake_embedder):
    index_dir = tmp_path / "index"
    ingest(sample_pdf, fake_embedder, index_dir, chunk_size=200, chunk_overlap=0)
    indexes = load_all_indexes(index_dir)
    chunks = indexes[0].chunks
    snow = next(c for c in chunks if "雪国" in c.text)
    cat = next(c for c in chunks if "猫" in c.text)

    cases = [
        EvalCase(question=f"SIMILAR_TO:{snow.text}", expected_pages=[snow.page]),
        EvalCase(question=f"SIMILAR_TO:{cat.text}", expected_pages=[cat.page]),
        EvalCase(question="全く関係のない質問", expected_pages=[999]),
    ]
    results = evaluate(cases, fake_embedder, indexes, top_k=3, min_score=0.0)
    assert results[0].top1_hit and results[1].top1_hit and not results[2].top3_hit
    assert all(r.search_time_sec >= 0 for r in results)

    report = build_report(results, indexes, 3, 0.0, tmp_path / "ds.json")
    assert report["model"] == "fake-embedder"
    assert report["chunk_size"] == 200 and report["top_k"] == 3
    assert report["summary"]["total"] == 3
    assert report["summary"]["top1_correct"] == 2
    assert report["summary"]["top3_correct"] == 2
    assert len(report["results"]) == 3
    assert "evaluated_at" in report


def test_evaluate_requires_top_k_at_least_3(fake_embedder, sample_pdf: Path, tmp_path: Path):
    index_dir = tmp_path / "index"
    ingest(sample_pdf, fake_embedder, index_dir)
    with pytest.raises(BookRagError, match="3 以上"):
        evaluate([EvalCase("q", [1])], fake_embedder, load_all_indexes(index_dir), top_k=2)


def test_main_missing_dataset(tmp_path: Path, capsys):
    rc = main(["--dataset", str(tmp_path / "none.json"), "--index-dir", str(tmp_path), "--output", str(tmp_path / "o.json")])
    assert rc == 1
    assert "見つかりません" in capsys.readouterr().err
