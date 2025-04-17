FROM python:3.10-slim

# 環境変数の設定（不要なら削除してOK）
# ENV PYTHONDONTWRITEBYTECODE=1
# ENV PYTHONUNBUFFERED=1

# 作業ディレクトリの作成と指定
WORKDIR /app

# pipと依存ライブラリのアップグレード
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# huggingface_hubによるモデルダウンロード（不要ファイル除外付き）
RUN python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download(repo_id='franciszzj/Leffa', local_dir='./ckpts', \
ignore_patterns=['*virtual_tryon.pth', '*pose_transfer.pth', 'stable-diffusion-xl-*'])"

# アプリのコードをコピー（server.pyなど）
COPY . .

# ポート（必要に応じて変更）
# EXPOSE 7860

# サーバ実行（例: Gradioアプリ）
CMD ["python", "server.py"]

