from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfgen import canvas

FONT = "HeiseiMin-W3"


class FakeEmbedder:
    """モデルをダウンロードせずにテストするためのダミー Embedder.

    テキストのハッシュから決定的な疑似ベクトルを作る。同じ文字列は同じベクトルになり、
    `similar_to` で「あるテキストに近いクエリ」を意図的に作れる。
    """

    model_name = "fake-embedder"
    dim = 16

    def _vec(self, text: str) -> np.ndarray:
        seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "little")
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(self.dim).astype(np.float32)
        return v / np.linalg.norm(v)

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._vec(t) for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        # "SIMILAR_TO:<passage>" 形式のクエリは、その passage と近いベクトルを返す
        if text.startswith("SIMILAR_TO:"):
            base = self._vec(text[len("SIMILAR_TO:") :])
            noise = self._vec("noise:" + text) * 0.05
            v = base + noise
            return (v / np.linalg.norm(v)).astype(np.float32)
        return self._vec(text)


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


def make_pdf(path: Path, pages: list[str | None]) -> Path:
    """pages の各要素を1ページとして PDF を作る. None は空ページ."""
    pdfmetrics.registerFont(UnicodeCIDFont(FONT))
    c = canvas.Canvas(str(path), pagesize=A4)
    for text in pages:
        if text:
            c.setFont(FONT, 11)
            y = 800
            for line in text.split("\n"):
                c.drawString(50, y, line)
                y -= 16
        c.showPage()
    c.save()
    return path


@pytest.fixture
def sample_pdf(tmp_path: Path) -> Path:
    return make_pdf(
        tmp_path / "sample.pdf",
        [
            "第一章。吾輩は猫である。名前はまだ無い。どこで生れたかとんと見当がつかぬ。",
            None,  # 空ページ
            "第二章。メロスは激怒した。必ず、かの邪智暴虐の王を除かなければならぬと決意した。",
            "第三章。国境の長いトンネルを抜けると雪国であった。夜の底が白くなった。",
        ],
    )
