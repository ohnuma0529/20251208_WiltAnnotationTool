# Wilt Annotation Tool (20251208 Version)

しおれ（Wilt）葉のアノテーションを行うためのWebベースツールです。
CoTracker3とSAM 2を使用して、キーフレーム間の自動追跡と補間を行い、高精度なアノテーションを効率的に作成できます。

## 特徴

*   **自動追跡 (Tracking)**: CoTracker3により、指定したキーポイント（Base, Tip）を映像全体で追跡します。
*   **領域分割 (Segmentation)**: SAM 2 (Segment Anything Model 2) を利用して、キーポイントから葉の領域（マスク）を自動生成します。
*   **高速レスポンス (Smart UI)**: 削除や修正などの操作が即座に画面に反映され、ストレスのないアノテーション作業が可能です。
*   **堅牢なデータ保存**: 追跡中や操作ごとの自動保存・バックアップ機能により、データ損失のリスクを最小限に抑えます。
*   **補正・編集**: キーポイントやBBoxを修正すると、即座に追跡結果とマスクが再計算されます。
*   **エクスポート**: YOLOフォーマットおよび詳細なCSV形式（正規化座標・画像サイズ付）でのデータ出力に対応しています。
*   **Systemd対応**: サーバー再起動時などに自動起動する設定を含んでいます。

## 必要要件

*   Linux (Ubuntu 推奨)
*   NVIDIA GPU (CUDA対応) - 必須 (CoTracker/SAM2用)
*   Anaconda / Miniconda
*   Node.js (Frontend用)

## セットアップ手順

### 1. リポジトリのクローンと環境構築

```bash
git clone https://github.com/ohnuma0529/20251208_WiltAnnotationTool.git
cd 20251208_WiltAnnotationTool

# Conda環境の作成 (環境名: WiltAnnotation)
conda env create -f environment.yml
conda activate WiltAnnotation
```

### 2. チェックポイントの準備

以下のモデルファイルを `checkpoints/` ディレクトリに配置してください。
(自動ダウンロードスクリプトが含まれていますが、手動で確認することを推奨します)

*   `sam2_hiera_large.pt`
*   `cotracker3_offline.pth` (Torch Hubキャッシュを利用する場合もあり)

### 3. 自動起動設定 (Systemd)

サーバー起動時や再起動時に自動的にシステムが立ち上がるように設定します。

```bash
# セットアップスクリプトの実行 (sudo権限が必要です)
bash setup_systemd.sh
```

これにより、以下のサービスが登録・起動されます。
*   `wilt-backend.service`: バックエンド (Port 8001)
*   `wilt-frontend.service`: フロントエンド (Port 5173)

### 手動起動 (開発用)

自動起動を利用せず、手動で起動する場合は以下のスクリプトを使用してください。

**バックエンド (Terminal 1)**
```bash
bash start_backend.sh
```

**フロントエンド (Terminal 2)**
```bash
bash start_frontend.sh
```

## 使い方

1.  ブラウザで `http://<サーバーIP>:5173` にアクセスします。
2.  **Unit** (個体) と **Date** (日付) を選択します。
3.  **Frequency** (表示間隔) を選択します (デフォルト: 30分)。
4.  **アノテーション**:
    *   画面上の葉に対して、根元(Base)から先端(Tip)へドラッグしてアノテーションを作成します。
    *   自動的に追跡が開始され、全フレームにアノテーションが伝播します。
5.  **補正**:
    *   追跡がズレているフレームで、キーポイントをドラッグして修正します (再追跡・再補間が行われます)。
    *   **Delete Future Images**: 選択フレーム以降の画像ソースを削除します（不可逆操作）。
    *   **Delete Leaf**: 選択した葉のアノテーションを削除します（即時反映）。
6.  **エクスポート**:
    *   **CSV Export**: 全フレーム（Freq 1分間隔）の詳細データをCSVとしてダウンロードします。これには正規化されたBBox、キーポイント座標、画像サイズ、手動修正フラグが含まれます。
    *   **YOLO Export**: YOLO学習用フォーマットでデータをエクスポートします。

## ディレクトリ構造

*   `backend/`: FastAPIサーバー、追跡ロジック (CoTracker/SAM2)
*   `frontend/`: React + Vite アプリケーション
*   `scripts/`: ユーティリティスクリプト
*   `systemd/`: 自動起動用サービス定義ファイル
*   `setup_systemd.sh`: Systemd登録スクリプト

## 注意事項

*   **GPUメモリ**: CoTrackerとSAM2を同時に動かすため、十分なVRAM (16GB以上推奨) が必要です。
*   **画像パス**: `backend/config.py` または `.env` (もしあれば) で画像ディレクトリのパスを確認してください。デフォルトは `/media/HDD-6TB/Leaf_Images` 等に設定されている場合があります。
