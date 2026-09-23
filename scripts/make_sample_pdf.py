"""青空文庫のテキストから評価用サンプルPDF（data/input/sample_aozora.pdf）を生成する.

対象: 太宰治『走れメロス』（著作権保護期間満了・パブリックドメイン）
usage: python scripts/make_sample_pdf.py [--output data/input/sample_aozora.pdf]

ネットワークから青空文庫の zip を取得し、ルビ・注記を除去して1ページあたり
一定行数で PDF 化する。生成物はテスト・評価用の同梱データ。
"""

from __future__ import annotations

import argparse
import io
import re
import sys
import urllib.request
import zipfile
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfgen import canvas

AOZORA_ZIP_URL = "https://www.aozora.gr.jp/cards/000035/files/1567_ruby_4948.zip"
AOZORA_TXT_NAME = "hashire_merosu.txt"
FONT = "HeiseiMin-W3"
FONT_SIZE = 10.5
LINE_HEIGHT = 17
CHARS_PER_LINE = 42
LINES_PER_PAGE = 36
MARGIN_X = 50
MARGIN_TOP = 800


def fetch_aozora_text() -> str:
    with urllib.request.urlopen(AOZORA_ZIP_URL, timeout=30) as res:
        data = res.read()
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        raw = zf.read(AOZORA_TXT_NAME)
    return raw.decode("shift_jis", errors="ignore")


def clean_aozora(text: str) -> str:
    """ルビ・注記・ヘッダ/フッタを除去して本文のみ返す."""
    text = text.replace("\r\n", "\n")
    # ヘッダ（区切り線で囲まれた凡例）を除去
    parts = text.split("-------------------------------------------------------")
    body = parts[-1] if len(parts) >= 3 else text
    # フッタ（底本情報）を除去
    body = body.split("底本：")[0]
    body = re.sub(r"《[^》]*》", "", body)  # ルビ
    body = body.replace("｜", "")  # ルビ開始記号
    body = re.sub(r"［＃[^］]*］", "", body)  # 注記
    return body.strip()


def wrap_lines(text: str) -> list[str]:
    lines: list[str] = []
    for paragraph in text.split("\n"):
        if not paragraph:
            lines.append("")
            continue
        for i in range(0, len(paragraph), CHARS_PER_LINE):
            lines.append(paragraph[i : i + CHARS_PER_LINE])
    return lines


def write_pdf(title: str, author: str, lines: list[str], output: Path) -> int:
    pdfmetrics.registerFont(UnicodeCIDFont(FONT))
    output.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(output), pagesize=A4)
    c.setTitle(title)
    c.setAuthor(author)

    pages = 0
    # 表紙
    c.setFont(FONT, 24)
    c.drawString(MARGIN_X, 600, title)
    c.setFont(FONT, 14)
    c.drawString(MARGIN_X, 560, author)
    c.setFont(FONT, FONT_SIZE)
    c.drawString(MARGIN_X, 520, "底本: 青空文庫（パブリックドメイン） / 書籍RAG MVP 評価用サンプル")
    c.showPage()
    pages += 1

    # 空ページ（除外処理の確認用）
    c.showPage()
    pages += 1

    for start in range(0, len(lines), LINES_PER_PAGE):
        c.setFont(FONT, FONT_SIZE)
        y = MARGIN_TOP
        for line in lines[start : start + LINES_PER_PAGE]:
            c.drawString(MARGIN_X, y, line)
            y -= LINE_HEIGHT
        pages += 1
        c.setFont(FONT, 9)
        c.drawCentredString(A4[0] / 2, 30, f"- {pages} -")
        c.showPage()
    c.save()
    return pages


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=Path("data/input/sample_aozora.pdf"))
    args = p.parse_args(argv)
    try:
        text = fetch_aozora_text()
    except Exception as e:  # noqa: BLE001
        print(f"エラー: 青空文庫からの取得に失敗しました: {e}", file=sys.stderr)
        return 1
    body = clean_aozora(text)
    pages = write_pdf("走れメロス", "太宰治", wrap_lines(body), args.output)
    print(f"生成しました: {args.output} ({pages}ページ, 本文 {len(body)} 文字)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
