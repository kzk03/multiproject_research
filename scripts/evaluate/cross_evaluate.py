#!/usr/bin/env python3
"""
============================================================
クロス評価スクリプト - 訓練期間×評価期間の性能評価
============================================================

【目的】
異なる訓練期間と評価期間の組み合わせで、モデルの性能を評価します。
これにより、以下を確認できます:
- 訓練に使う将来窓の長さがモデル性能に与える影響
- 評価に使う将来窓の長さが予測難易度に与える影響
- 最適な訓練・評価期間の組み合わせ

【クロス評価の仕組み】
訓練期間（将来窓）: 0-3m, 3-6m, 6-9m, 9-12m
評価期間（将来窓）: 0-3m, 3-6m, 6-9m, 9-12m

合計: 4 × 4 = 16 通りの組み合わせ

例:
- train_0-3m / eval_0-3m: 直近0-3ヶ月の継続で訓練 → 直近0-3ヶ月の継続を予測
- train_0-3m / eval_9-12m: 直近0-3ヶ月の継続で訓練 → 9-12ヶ月後の継続を予測

【実行フロー】
1. 各訓練期間でモデルを訓練（4モデル）
   - train_0-3m: 訓練期間の将来窓を0-3ヶ月に設定
   - train_3-6m: 訓練期間の将来窓を3-6ヶ月に設定
   - train_6-9m: 訓練期間の将来窓を6-9ヶ月に設定
   - train_9-12m: 訓練期間の将来窓を9-12ヶ月に設定

2. 各モデルで全評価期間を評価（4×4=16評価）
   - 各訓練モデルを読み込み
   - 評価期間の将来窓を変えて4回評価
   - メトリクス（AUC-ROC、AUC-PR、F1など）を保存

【出力構造】
outputs/cross_eval/
├── train_0-3m/
│   ├── irl_model.pt            # 訓練されたモデル
│   ├── eval_0-3m/metrics.json  # 評価結果
│   ├── eval_3-6m/metrics.json
│   ├── eval_6-9m/metrics.json
│   └── eval_9-12m/metrics.json
├── train_3-6m/
│   └── ...
├── train_6-9m/
│   └── ...
└── train_9-12m/
    └── ...

【使用方法】
直接実行:
    uv run python scripts/evaluate/cross_evaluate.py

結果の可視化:
    uv run python scripts/evaluate/create_heatmaps.py
"""
import logging
import subprocess
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ========================================
# 設定: ファイルパスと期間定義
# ========================================
BASE_OUTPUT_DIR = Path("outputs/cross_eval")
TRAIN_SCRIPT = "scripts/train/train_model.py"
REVIEWS_DATA = "data/review_requests_openstack_multi_5y_detail.csv"

# 訓練期間と評価期間の定義（将来窓）
train_periods = ['0-3m', '3-6m', '6-9m', '9-12m']
eval_periods = ['0-3m', '3-6m', '6-9m', '9-12m']

# 基準日
TRAIN_START = "2021-01-01"
TRAIN_END = "2023-01-01"
EVAL_START = "2023-01-01"
EVAL_END = "2024-01-01"

def get_month_offset(period_str):
    """期間文字列から月数オフセットを取得"""
    start, end = period_str.split('-')
    start_month = int(start.replace('m', ''))
    end_month = int(end.replace('m', ''))
    return start_month, end_month

# 1. 訓練期間ごとにモデルを訓練
logger.info("=" * 80)
logger.info("訓練期間ごとにモデルを訓練")
logger.info("=" * 80)

for train_period in train_periods:
    train_start_month, train_end_month = get_month_offset(train_period)
    
    output_dir = BASE_OUTPUT_DIR / f"train_{train_period}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = output_dir / "irl_model.pt"
    
    if model_path.exists():
        logger.info(f"スキップ: train_{train_period} (モデル既に存在)")
        continue
    
    logger.info(f"訓練中: train_{train_period}")
    
    # 訓練期間の評価も同時に実行（0-3mの評価）
    command = [
        "uv", "run", "python", TRAIN_SCRIPT,
        "--reviews", REVIEWS_DATA,
        "--train-start", TRAIN_START,
        "--train-end", TRAIN_END,
        "--eval-start", EVAL_START,
        "--eval-end", EVAL_END,
        "--future-window-start", str(train_start_month),
        "--future-window-end", str(train_end_month),
        "--epochs", "30",  # エポック数を30に設定
        "--min-history-events", "3",
        "--output", str(output_dir),
        "--project", "openstack/nova"
    ]
    
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        logger.info(f"成功: train_{train_period}")
    except subprocess.CalledProcessError as e:
        logger.error(f"エラー: {e.stderr}")
        logger.error(f"失敗: train_{train_period}")
        continue

# 2. 各訓練モデルで全評価期間を評価
logger.info("=" * 80)
logger.info("クロス評価を実行")
logger.info("=" * 80)

for train_period in train_periods:
    train_model_path = BASE_OUTPUT_DIR / f"train_{train_period}" / "irl_model.pt"
    
    if not train_model_path.exists():
        logger.warning(f"モデルが存在しません: {train_model_path}")
        continue
    
    for eval_period in eval_periods:
        eval_start_month, eval_end_month = get_month_offset(eval_period)
        
        output_dir = BASE_OUTPUT_DIR / f"train_{train_period}" / f"eval_{eval_period}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        metrics_file = output_dir / "metrics.json"
        
        if metrics_file.exists():
            logger.info(f"スキップ: train_{train_period} -> eval_{eval_period} (既に存在)")
            continue
        
        logger.info(f"評価中: train_{train_period} -> eval_{eval_period}")
        
        command = [
            "uv", "run", "python", TRAIN_SCRIPT,
            "--reviews", REVIEWS_DATA,
            "--train-start", TRAIN_START,
            "--train-end", TRAIN_END,
            "--eval-start", EVAL_START,
            "--eval-end", EVAL_END,
            "--future-window-start", str(eval_start_month),
            "--future-window-end", str(eval_end_month),
            "--epochs", "20",
            "--min-history-events", "3",
            "--output", str(output_dir),
            "--project", "openstack/nova",
            "--model", str(train_model_path)
        ]
        
        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            logger.info(f"成功: train_{train_period} -> eval_{eval_period}")
        except subprocess.CalledProcessError as e:
            logger.error(f"エラー: {e.stderr}")
            logger.error(f"失敗: train_{train_period} -> eval_{eval_period}")
            continue

logger.info("=" * 80)
logger.info("全てのクロス評価が完了しました！")
logger.info("=" * 80)

