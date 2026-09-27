"""画像のみのPDFページを Tesseract で OCR する.

pypdfium2 でページを画像化し、pytesseract で文字認識する。
Tesseract 本体と日本語データ（jpn / jpn_vert）は別途インストールが必要（README 参照）。
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

from .common import BookRagError, logger

OCR_MODES = ("auto", "off", "force")
OCR_LANGS = ("auto", "jpn", "jpn_vert")
DEFAULT_OCR_MODE = "auto"
DEFAULT_OCR_LANG = "auto"
DEFAULT_OCR_DPI = 300
_MAX_RENDER_PX = 3500  # 長辺の上限ピクセル（スマホ撮影の巨大ページ対策）

_JA_CHARS = re.compile(r"[\u3040-\u30ff\u4e00-\u9fff]")
# Tesseract の日本語出力は文字間に空白が入るため、全角文字同士の間の空白は除去する
_SPACE_BETWEEN_JA = re.compile(r"(?<=[^\x00-\x7F])[ \t]+(?=[^\x00-\x7F])")


def check_tesseract(lang: str) -> None:
    """Tesseract 本体と言語データの有無を確認し、無ければ導入方法付きでエラーにする."""
    import pytesseract

    if shutil.which("tesseract") is None:
        raise BookRagError(
            "OCR には Tesseract が必要です。以下でインストールしてください:\n"
            "  macOS : brew install tesseract tesseract-lang\n"
            "  Ubuntu: sudo apt install tesseract-ocr tesseract-ocr-jpn tesseract-ocr-jpn-vert\n"
            "OCR を使わない場合は --ocr off を指定してください。"
        )
    available = set(pytesseract.get_languages())
    required = {"jpn", "jpn_vert"} if lang == "auto" else {lang}
    missing = sorted(required - available)
    if missing:
        raise BookRagError(
            f"Tesseract の言語データがありません: {', '.join(missing)}\n"
            "  macOS : brew install tesseract-lang\n"
            "  Ubuntu: sudo apt install tesseract-ocr-jpn tesseract-ocr-jpn-vert"
        )


def clean_ocr_text(text: str) -> str:
    text = _SPACE_BETWEEN_JA.sub("", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    return text.strip()


def ja_char_count(text: str) -> int:
    return len(_JA_CHARS.findall(text))


class PageOcr:
    """1つのPDFに対する OCR 器. 言語 auto の場合は最初のページで横書き/縦書きを判定する."""

    def __init__(self, pdf_path: Path, lang: str = DEFAULT_OCR_LANG, dpi: int = DEFAULT_OCR_DPI):
        if lang not in OCR_LANGS:
            raise BookRagError(f"--ocr-lang は {', '.join(OCR_LANGS)} のいずれかを指定してください")
        check_tesseract(lang)
        import pypdfium2 as pdfium
        import pytesseract

        self._doc = pdfium.PdfDocument(str(pdf_path))
        self._dpi = dpi
        self._osd_available = "osd" in pytesseract.get_languages()
        if not self._osd_available:
            logger.warning("Tesseract の osd データが無いため、ページの向き自動補正は行いません")
        self.lang: str | None = None if lang == "auto" else lang

    def _render(self, page_no: int):
        page = self._doc[page_no - 1]
        width, height = page.get_size()
        scale = min(self._dpi / 72, _MAX_RENDER_PX / max(width, height))
        return page.render(scale=scale).to_pil().convert("L")

    @staticmethod
    def _run(image, lang: str) -> str:
        import pytesseract

        return clean_ocr_text(pytesseract.image_to_string(image, lang=lang, config="--psm 3"))

    def _fix_rotation(self, image, page_no: int):
        """スマホ写真で横向きになったページを Tesseract の向き判定（OSD）で起こす."""
        import pytesseract

        if not self._osd_available:
            return image
        try:
            osd = pytesseract.image_to_osd(image, config="--psm 0")
        except pytesseract.TesseractError as e:
            logger.debug("ページ %d の向き判定に失敗: %s", page_no, e)
            return image
        m = re.search(r"^Rotate:\s*(\d+)", osd, re.MULTILINE)
        rotate = int(m.group(1)) if m else 0
        if rotate:
            logger.info("ページ %d は %d度回転して OCR します", page_no, rotate)
            # Tesseract の Rotate は時計回りでの補正角。PIL.rotate は反時計回り
            return image.rotate(-rotate, expand=True)
        return image

    def ocr_page(self, page_no: int) -> str:
        image = self._fix_rotation(self._render(page_no), page_no)
        if self.lang is None:
            candidates = {lang: self._run(image, lang) for lang in ("jpn", "jpn_vert")}
            self.lang = max(candidates, key=lambda k: ja_char_count(candidates[k]))
            logger.info(
                "OCR 言語を自動判定: %s (jpn=%d文字, jpn_vert=%d文字)",
                self.lang,
                ja_char_count(candidates["jpn"]),
                ja_char_count(candidates["jpn_vert"]),
            )
            return candidates[self.lang]
        return self._run(image, self.lang)
