"""画像のみのPDFページを OCR する.

pypdfium2 でページを画像化し、以下いずれかのエンジンで文字認識する。
- vision   : macOS 標準の Vision フレームワーク（ocrmac 経由）。日本語精度が高く速い。macOS 専用
- tesseract: Tesseract（pytesseract 経由）。本体と日本語データ（jpn / jpn_vert）を別途インストール
engine=auto の場合は macOS で ocrmac が使えれば vision、それ以外は tesseract を使う。
"""

from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

from .common import BookRagError, logger

OCR_MODES = ("auto", "off", "force")
OCR_LANGS = ("auto", "jpn", "jpn_vert")
OCR_ENGINES = ("auto", "vision", "tesseract")
DEFAULT_OCR_MODE = "auto"
DEFAULT_OCR_LANG = "auto"
DEFAULT_OCR_ENGINE = "auto"
DEFAULT_OCR_DPI = 300
_MAX_RENDER_PX = 3500  # 長辺の上限ピクセル（スマホ撮影の巨大ページ対策）
_VISION_MIN_JA_CHARS = 20  # これ未満なら向きが違う可能性があるので回転して再試行

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


def vision_available() -> bool:
    if sys.platform != "darwin":
        return False
    try:
        import ocrmac  # noqa: F401
    except ImportError:
        return False
    return True


def clean_ocr_text(text: str) -> str:
    text = _SPACE_BETWEEN_JA.sub("", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    return text.strip()


def ja_char_count(text: str) -> int:
    return len(_JA_CHARS.findall(text))


def order_vision_lines(lines: list[tuple[str, float, list[float]]]) -> list[str]:
    """Vision の認識結果（行, 信頼度, [x, y, w, h]）を読み順に並べる.

    座標は 0〜1 の正規化値で原点は左下。縦長の行が多ければ縦書きとみなし右→左、
    それ以外は上→下・左→右に並べる。
    """
    if not lines:
        return []
    vertical = sum(1 for _, _, (_, _, w, h) in lines if h > w * 1.5) > len(lines) / 2
    if vertical:
        ordered = sorted(lines, key=lambda r: (-(r[2][0] + r[2][2]), -(r[2][1] + r[2][3])))
    else:
        ordered = sorted(lines, key=lambda r: (-round(r[2][1] + r[2][3], 2), r[2][0]))
    return [text for text, _, _ in ordered]


class _PageRenderer:
    def __init__(self, pdf_path: Path, dpi: int):
        import pypdfium2 as pdfium

        self._doc = pdfium.PdfDocument(str(pdf_path))
        self._dpi = dpi

    def render(self, page_no: int):
        page = self._doc[page_no - 1]
        width, height = page.get_size()
        scale = min(self._dpi / 72, _MAX_RENDER_PX / max(width, height))
        return page.render(scale=scale).to_pil().convert("L")


class TesseractPageOcr:
    """Tesseract による OCR 器. 言語 auto の場合は最初のページで横書き/縦書きを判定する."""

    def __init__(self, pdf_path: Path, lang: str = DEFAULT_OCR_LANG, dpi: int = DEFAULT_OCR_DPI):
        if lang not in OCR_LANGS:
            raise BookRagError(f"--ocr-lang は {', '.join(OCR_LANGS)} のいずれかを指定してください")
        check_tesseract(lang)
        import pytesseract

        self._renderer = _PageRenderer(pdf_path, dpi)
        self._osd_available = "osd" in pytesseract.get_languages()
        if not self._osd_available:
            logger.warning("Tesseract の osd データが無いため、ページの向き自動補正は行いません")
        self.lang: str | None = None if lang == "auto" else lang

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
        image = self._fix_rotation(self._renderer.render(page_no), page_no)
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


class VisionPageOcr:
    """macOS Vision フレームワーク（ocrmac）による OCR 器. 横書き/縦書きは Vision 側が自動判定する."""

    def __init__(self, pdf_path: Path, lang: str = DEFAULT_OCR_LANG, dpi: int = DEFAULT_OCR_DPI):
        if lang not in OCR_LANGS:
            raise BookRagError(f"--ocr-lang は {', '.join(OCR_LANGS)} のいずれかを指定してください")
        if not vision_available():
            raise BookRagError(
                "Vision OCR は macOS 専用で、ocrmac が必要です: pip install ocrmac\n"
                "Tesseract を使う場合は --ocr-engine tesseract を指定してください。"
            )
        self._renderer = _PageRenderer(pdf_path, dpi)

    @staticmethod
    def _run(image) -> tuple[str, int]:
        from ocrmac import ocrmac

        try:
            lines = ocrmac.OCR(image, language_preference=["ja-JP"]).recognize()
        except ValueError as e:
            raise BookRagError(f"Vision OCR で日本語(ja-JP)が使えません: {e}") from e
        text = clean_ocr_text("\n".join(order_vision_lines(lines)))
        return text, ja_char_count(text)

    def ocr_page(self, page_no: int) -> str:
        image = self._renderer.render(page_no)
        text, count = self._run(image)
        if count >= _VISION_MIN_JA_CHARS:
            return text
        # 文字がほとんど取れない場合は横向き/逆さの可能性があるので回転して最良を採用
        best_text, best_count, best_angle = text, count, 0
        for angle in (90, 180, 270):
            t, c = self._run(image.rotate(angle, expand=True))
            if c > best_count:
                best_text, best_count, best_angle = t, c, angle
        if best_angle:
            logger.info("ページ %d は %d度回転して OCR しました", page_no, best_angle)
        return best_text


def create_page_ocr(
    pdf_path: Path,
    lang: str = DEFAULT_OCR_LANG,
    engine: str = DEFAULT_OCR_ENGINE,
) -> TesseractPageOcr | VisionPageOcr:
    if engine not in OCR_ENGINES:
        raise BookRagError(f"--ocr-engine は {', '.join(OCR_ENGINES)} のいずれかを指定してください")
    if engine == "vision":
        return VisionPageOcr(pdf_path, lang)
    if engine == "tesseract":
        return TesseractPageOcr(pdf_path, lang)
    if vision_available():
        logger.info("OCR エンジン: Vision (macOS)")
        return VisionPageOcr(pdf_path, lang)
    if sys.platform == "darwin":
        logger.warning("ocrmac が未インストールのため Tesseract を使います（pip install ocrmac で高精度化）")
    logger.info("OCR エンジン: Tesseract")
    return TesseractPageOcr(pdf_path, lang)
