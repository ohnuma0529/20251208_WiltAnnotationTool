# WiltAnnotationTool (トマト萎凋病アノテーションツール)

SAM 2 (Segment Anything Model 2) と CoTracker を活用した、植物（トマトの葉）の萎凋病進行を高精度にアノテーション・追跡するためのツールです。

## 主な特徴 (Features)

### 1. 高度なトラッキング機能
- **Dense Tracking**: 全フレーム（Frequency=1相当）に対して高密度なトラッキング処理をバックグラウンドで実行し、指定した頻度（Frequency=30分など）で表示します。
- **Tight BBox ("Bitabita")**: 葉のポイント（Base/Tip）とサポートポイントを厳密に囲むバウンディングボックスを自動生成します（+10pxのパディング付き）。
- **Outlier Filtering**: トラッキング中に発生した外れ値（誤検出されたサポートポイント）を統計的に除去し、BBoxの精度を維持します。

### 2. 直感的なUI/UX
- **Frequency Control**: タイムラインの表示頻度を柔軟に変更可能（例: 1分, 10分, 30分間隔）。
- **Sync BBox**: キーポイント（Base/Tip）をドラッグして修正すると、BBoxも自動的に追従して再計算されます。
- **Global Delete**: 特定の葉を「全てのフレーム」から一括削除する機能。
- **Optimized Performance**: 1200枚以上の画像をスムーズに扱えるキャッシュシステムとバックエンド最適化。

### 3. エクスポート
- **YOLO形式**: 学習用に正規化されたBBoxとキーポイントデータをZIPエクスポート。
- **CSV形式**: 詳細な解析用に、全フレーム（Dense）の座標データを含むCSVをエクスポート。

---

## セットアップ (Setup)

### 前提条件
- Linux (Ubuntu推奨)
- NVIDIA GPU (VRAM 8GB以上推奨, CUDA 12.1+)
- Anaconda / Miniconda

### 環境構築
付属のセットアップスクリプトを使用して環境を構築します。

```bash
bash setup_env.sh
```
これにより、conda環境 `WiltAnnotation` が作成され、必要なライブラリやモデル（SAM 2, CoTracker）がインストールされます。

---

## 実行方法 (Usage)

### 1. 環境の有効化
```bash
conda activate WiltAnnotation
```

### 2. バックエンドの起動
```bash
./start_backend.sh
```
ポート **8001** でAPIサーバーが起動します。

### 3. フロントエンドの起動
別のターミナルを開き、同様に環境を有効化してから実行します。
```bash
./start_frontend.sh
```
ブラウザが開き、ツールが表示されます（デフォルト: `http://localhost:5173`）。

---

## アノテーションの流れ (Workflow)

1.  **Unit/Date/Frequencyの選択**:
    サイドバーから対象の温室ユニット(Unit)、日付(Date)を選択し、表示頻度(Frequency)を設定します。

2.  **葉のアノテーション (Create Leaf)**:
    - 画像上の葉の **付け根 (Base)** と **先端 (Tip)** を順にクリックします。
    - クリックすると自動的にBBoxが生成され、SAM 2によるセグメンテーション（マスク生成）とサポートポイント生成が行われます。

3.  **トラッキング実行 (Track)**:
    - 「Start Tracking」ボタンを押すと、作成した葉のポイントが全期間にわたって追跡されます。
    - 処理はバックグラウンドで行われ、進捗バーが表示されます。

4.  **修正 (Correction)**:
    - トラッキング結果を確認し、ズレている場合はポイントをドラッグして修正します。
    - **Sync BBox**: ポイントを動かすとBBoxも自動修正されます。
    - 必要に応じて「Update BBox」で再計算させることも可能です。

5.  **エクスポート (Export)**:
    - 「Export YOLO」または「Export CSV」ボタンで、アノテーション結果をダウンロードします。

---

## ディレクトリ構成
- `backend/`: FastAPI + PyTorch (SAM2, CoTracker)
- `frontend/`: React + Vite + TailwindCSS
- `fast_cache/`: 画像の高速読み込み用キャッシュ（Git対象外）
- `images/`: データセット置き場（Git対象外）
