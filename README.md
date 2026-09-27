# book-rag — 書籍RAG検索 MVP

PDF形式の書籍を取り込み、質問文と意味的に近い本文を **出典ページ番号付き** で検索する最小構成のRAG検索基盤です。

- Embedding: [`intfloat/multilingual-e5-small`](https://huggingface.co/intfloat/multilingual-e5-small)（日本語対応・ローカル実行・API課金なし）
- ベクトルストア: NumPy ファイル（外部DB不要）
- スキャン画像 PDF（スマホで撮った本など）は OCR で取り込み可能（macOS: Vision / その他: Tesseract）
- CLI のみ（LLMによる回答生成 / Web UI / 認証は対象外）

## 動作環境

- Python 3.10 以上
- CPU のみで動作（初回のみ Embedding モデル約 470MB を Hugging Face からダウンロード）

## 1. 環境構築

```bash
git clone https://github.com/Yoshinana727/book-rag.git
cd book-rag
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# torch は CPU 版を先にインストール（Linux/Windows。GPU版は 2GB 超になるため）
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.0
# macOS の場合は index-url なしで: pip install torch==2.14.0
pip install -r requirements.txt
```

スキャン画像 PDF を OCR で取り込む場合は OCR エンジンを入れます（文字情報を持つ PDF だけを扱うなら不要）:

```bash
# macOS（推奨）: OS 標準の Vision フレームワークを使う。日本語の認識精度が高く高速
#   ※ requirements.txt に含まれているので macOS では pip install -r requirements.txt で導入済み
#   別途入れる場合: pip install ocrmac

# Linux など macOS 以外（もしくは macOS で Tesseract を使いたい場合）
sudo apt install tesseract-ocr tesseract-ocr-jpn tesseract-ocr-jpn-vert   # Ubuntu
brew install tesseract tesseract-lang                                      # macOS
tesseract --list-langs   # jpn / jpn_vert / osd が含まれていれば OK
```

動作確認（モデルのダウンロードなしで実行できます）:

```bash
python -m pytest
```

> `which python` / `python -m pip list` で venv 内の Python とパッケージが使われていることを確認してください。`ModuleNotFoundError: No module named 'numpy'` が出る場合は venv 外の `pytest` を実行しているか、`pip install` が失敗しています。

## 2. PDF の登録（ingest）

PDF を `data/input/` に置き、取り込みコマンドを実行します。`data/input/` と `data/index/` は `.gitignore` 済みで、著作権のある本文や生成インデックスが Git に入ることはありません。

```bash
cp /path/to/your_book.pdf data/input/
python -m src.ingest data/input/your_book.pdf
```

出力例:

```text
WARNING: ページ 2 は本文を抽出できなかったため除外します
INFO: 抽出完了: your_book.pdf 全10ページ / 処理9ページ / 除外1ページ
INFO: チャンク生成: 29件 (chunk_size=500, overlap=100)
INFO: インデックス保存先: .../data/index/your_book
=== 取り込み結果 ===
書籍名      : your_book.pdf
全ページ数  : 10
処理ページ数: 9
除外ページ数: 1 [2]
OCRページ数  : 0
生成チャンク: 29
モデル      : intfloat/multilingual-e5-small
```

| オプション | 既定値 | 説明 |
|---|---|---|
| `--chunk-size` | 500 | チャンクの文字数。文末（。！？改行）を優先して区切る |
| `--chunk-overlap` | 100 | 隣接チャンク間の重複文字数 |
| `--model` | `intfloat/multilingual-e5-small` | Embedding モデル（Hugging Face のモデル名） |
| `--index-dir` | `data/index` | インデックス保存先 |
| `--ocr` | `auto` | `auto`: 文字が取れない画像ページのみ OCR / `off`: OCR しない / `force`: 全ページ OCR |
| `--ocr-engine` | `auto` | `auto`: macOS で ocrmac があれば `vision`、無ければ `tesseract` |
| `--ocr-lang` | `auto` | Tesseract 用。`auto`: 最初の OCR ページで横書き(`jpn`)/縦書き(`jpn_vert`)を自動判定（Vision では無視） |

### スキャン画像 PDF（OCR）

iPhone のメモ/ファイルの「書類をスキャン」などで作った PDF は文字情報を持たない画像のみの PDF です。`--ocr auto`（既定）ではこうしたページを自動で OCR します（OCR エンジンが未導入の場合は警告を出して除外）。

- OCR エンジンは自動選択されます（ログに `OCR エンジン: Vision (macOS)` / `Tesseract` と出ます）。macOS では Vision の方が誤字が大幅に少なく高速なのでこちらを推奨。比較したい場合は `--ocr-engine tesseract` で強制できます
- Tesseract は 1ページあたり数秒かかります（300ページで 15〜30 分程度）。Vision はその数分の一
- 横向きに撮ったページは自動で回転補正します（Tesseract は `osd` データが必要。Vision は文字がほとんど取れない場合に 90/180/270 度回転して再試行）
- エンジンを変えた場合は再度 `python -m src.ingest` を実行してインデックスを作り直してください
- 認識精度を上げるコツ: 本を平らに開き、**1ページごとに**、ページ全体が真っ直ぐ入るように撮影する。見開きで撮ると左右の本文が混ざり、出典ページもずれます
- スキャン設定は「白黒」・「グレースケール」にするとファイルサイズが大幅に小さくなり、転送も速くなります
- すでに文字情報を持つ PDF（Adobe Scan の OCR 済みなど）はそのまま読み込まれ、OCR は走りません

生成物（`data/index/<書籍名>/`）:

- `chunks.json` — `{"book", "page", "chunk_id", "text"}` のチャンク一覧
- `embeddings.npy` — チャンクのベクトル（L2 正規化済み float32）
- `meta.json` — **実験条件の記録**（モデル名・chunk_size・overlap・作成日時・処理/除外ページ数・チャンク数）

複数の PDF を取り込むと書籍ごとにサブディレクトリが作られ、検索時はすべて横断検索されます。

### サンプルPDF

評価・動作確認用に、青空文庫（パブリックドメイン）の太宰治『走れメロス』から生成した `data/input/sample_aozora.pdf`（10ページ、2ページ目は意図的な空ページ）を同梱しています。再生成する場合:

```bash
python scripts/make_sample_pdf.py
```

## 3. 検索（search）

```bash
python -m src.search "王が人を信じられなくなった理由は何か"
```

出力例:

```text
[1] sample_aozora.pdf / 3ページ / score: 0.85
楽しみである。歩いているうちにメロスは、まちの様子を怪しく思った。…「王様は、人を殺します。」「なぜ殺すのだ。」…

[2] sample_aozora.pdf / 4ページ / score: 0.85
…
```

| オプション | 既定値 | 説明 |
|---|---|---|
| `--top-k` | 3 | 表示件数 |
| `--min-score` | 0.75 | この類似度未満の結果は表示しない |
| `--book` | (全書籍) | 検索対象を特定の PDF ファイル名に絞る |
| `--json` | — | 結果を JSON で出力 |
| `--model` | インデックス作成時のモデル | 別モデルを指定すると警告が出ます |

該当がない、または全件が `--min-score` 未満の場合:

```text
該当する本文が見つかりませんでした（類似度が基準値 0.75 を下回りました）
```

> **min-score について**: e5 系モデルはコサイン類似度が 0.7〜0.9 の狭い範囲に集まる性質があり、無関係な質問でも 0.75 前後になります。既定値 0.75 は目安なので、対象書籍に合わせて調整してください（`--min-score 0` でフィルタなし）。

## 4. 評価（evaluate）

質問と正解ページの組を `datasets/evaluation.json` に用意します（同梱データはサンプルPDF用に6件）。

```json
[
  {
    "question": "峠でメロスを襲ったのは誰か？",
    "expected_pages": [7],
    "note": "任意メモ"
  }
]
```

```bash
python -m src.evaluate
# 別のデータ / 出力先を使う場合
python -m src.evaluate --dataset datasets/my_book.json --book my_book.pdf --output results/my_book.json
```

出力例:

```text
=== 評価結果 ===
モデル        : intfloat/multilingual-e5-small
chunk_size    : 500 / overlap: 100 / top_k: 3
質問数        : 6
Top1 正解     : 4/6 (66.7%)
Top3 正解     : 6/6 (100.0%)
平均検索時間  : 0.008 秒/質問

--- Top3 で正解ページを取得できなかった質問 ---
Q: ...
   期待ページ: [7] / 取得ページ: [3, 4, 8] / score: [...]
```

`results/evaluation.json` に、評価日時・モデル名・chunk_size・overlap・top_k・各質問の結果（取得ページ・スコア・Top1/Top3 判定・検索時間）・集計値を保存します。`failure_analysis` は失敗原因の考察を手書きで追記する欄です。`results/` は Git 管理外です。

- Top1 正解: 1位のチャンクのページが `expected_pages` に含まれる
- Top3 正解: 上位3件のいずれかのページが `expected_pages` に含まれる
- 検索時間: 質問のベクトル化 + 類似度計算（モデルのロード時間は含まない）

## 5. 自分の書籍で試す手順

```bash
cp ~/Downloads/my_book.pdf data/input/
python -m src.ingest data/input/my_book.pdf          # 取り込み
python -m src.search "質問文" --book my_book.pdf      # 検索
# datasets/my_book.json に質問と正解ページを3件以上書く
python -m src.evaluate --dataset datasets/my_book.json --book my_book.pdf --output results/my_book.json
```

同じ条件で再実行したい場合は `data/index/<書籍名>/meta.json` と `results/*.json` に記録された `model` / `chunk_size` / `chunk_overlap` / `top_k` を `--model` / `--chunk-size` / `--chunk-overlap` / `--top-k` に指定してください。

## 6. トラブルシューティング

| 症状 | 原因・対処 |
|---|---|
| `エラー: PDFファイルが見つかりません` | パスを確認。`data/input/` からの相対パスまたは絶対パスを指定 |
| `エラー: 本文を1ページも抽出できませんでした` | スキャン画像 PDF の場合は Tesseract を導入して `--ocr auto` で実行。OCR 済みでも出る場合は画像の向き・解像度を確認 |
| `WARNING: OCR をスキップします: OCR には Tesseract が必要です` | macOS: `pip install ocrmac` / その他: Tesseract を導入（環境構築参照） |
| `WARNING: ocrmac が未インストールのため Tesseract を使います` | macOS で Vision を使うには `pip install ocrmac` |
| OCR の誤字が多い（Tesseract） | macOS なら `--ocr-engine vision` を使う。縦書き/横書きの誤判定の場合は `--ocr-lang jpn_vert` または `--ocr-lang jpn` を明示 |
| Vision で行の順序が乱れる | 縦書きは右→左、横書きは上→下に座標で並べ直しています。見開き撮影だと左右ページが混ざるので 1ページごとに撮影 |
| `エラー: 暗号化されたPDFは読み込めません` | パスワード保護を解除した PDF を用意 |
| `エラー: インデックスが1件もありません` | 先に `python -m src.ingest` を実行 |
| モデルのダウンロードに失敗する | ネットワーク／プロキシを確認。DL 済みモデルは `~/.cache/huggingface/` にキャッシュされ 2 回目以降はオフラインで動作 |
| `Warning: You are sending unauthenticated requests to the HF Hub` | 無視して問題なし（レート制限の案内） |
| 無関係な質問でも結果が表示される | `--min-score` を上げる（e5 系はスコアが高めに出ます） |

## ファイル構成

```text
book-rag/
├── src/
│   ├── common.py         # 定数・Embedder・インデックスの保存/読込
│   ├── ingest.py         # PDF抽出 → チャンク化 → ベクトル化 → 保存
│   ├── ocr.py            # スキャン画像ページの OCR（macOS Vision / Tesseract）
│   ├── search.py         # 質問のベクトル化 → 類似検索 → 表示
│   └── evaluate.py       # 評価データで精度測定 → results/ に保存
├── scripts/make_sample_pdf.py   # 青空文庫からサンプルPDFを生成
├── data/input/           # PDF置き場（sample_aozora.pdf 以外は Git 管理外）
├── data/index/           # 生成インデックス（Git 管理外）
├── datasets/evaluation.json     # 評価データ
├── results/              # 評価結果（Git 管理外）
├── tests/                # pytest（モデル不要のダミー Embedder で実行）
├── requirements.txt
└── README.md
```
