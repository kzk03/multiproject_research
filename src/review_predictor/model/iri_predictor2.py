import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

logger = logging.getLogger(__name__)

@dataclass
class DeveloperState:
    "開発者の状態表現(10次元ベクトル)"
    developer_id: str
    experimence_days: int
    total_changes: int
    total_reveiws: int
    recent_activity_frequency: float
    ave_activity_gap: float
    collaboration_trend: str
    code_quality_score: float
    recent_acceptance_rate: float
    review_load: float
    timestamp: datetime

@dataclass  
class DeveloperAction:
    "開発者の行動表現(5次元ベクトル)"
    action_type: str
    intensity: float
    collaboration_level: float
    response_time: float
    review_size: float
    timestamp: datetime

class RetentionIRLNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128, dropout: float = 0.1):
        # ネットワークの初期化
        super().__init__()
        # 状態エンコーダ
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), # 10->128
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2), # 128->64
            nn.Dropout(dropout)
        )
        # 行動エンコーダ
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim), # 5->128
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2), # 128->64
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        # LSTM
        self.lstm = nn.LSTM(
            input_size = hidden_dim // 2,
            hidden_size = hidden_dim,
            num_layers = 1 , # 増やしすぎると過学習のリスクあり（増やすことも検討できる）
            batch_first = True # 直感的な並び順で扱えるようにする
        )
        # 報酬予測器
        self.reward_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

        def forward(self, state: torch.Tensor, action: torch.Tensor, seq_lengths: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
            #順伝播（常にLSTM）
            batch_size, seq_len, _ = state.size()
            # 状態・行動のエンコード
            state_encoded = self.state_encoder(state.view(-1, state.shape[-1])).view(batch_size, seq_len, -1)
            action_encoded = self.action_encoder(action.view(-1, action.shape[-1])).view(batch_size, seq_len, -1)
            # 状態・行動の結合
            combined = state_encoded + action_encoded

            # 可変長シーケンスの処理
            if seq_lengths is not None:
                 # PyTorchのLSTMは降順にソートされたシーケンスを要求するため、ソートを行う
                lengths_cpu = seq_lengths.cpu()
                sorted_lengths, sorted_idx = lengths_cpu.sort(descending=True)
                _, unsort_idx = sorted_idx.sort()  # 元の順序に戻すためのインデックス
            # ソート後のシーケンスをpack
            combined_sorted = combined[sorted_idx]
            packed = nn.utils.rnn.pack_padded_sequence(
                combined_sorted, sorted_lengths, batch_first=True, enforce_sorted=True
            )
            # LSTMに入力
            lstm_out_packed, _ = self.lstm(packed)
            # unpack
            lstm_out_sorted, _ = nn.utils.rnn.pad_packed_sequence(
                lstm_out_packed, batch_first=True
            )
            lstm_out = lstm_out_sorted[unsort_idx]
            # 各シーケンスの実際の最終ステップを取得
            # パディング部分を無視し、実際のデータの最終ステップのみを使用
            hidden = torch.zeros(batch_size, lstm_out.size(-1), device=state.device)
            for i in range(batch_size):
                actual_len = lengths[i].item()
                hidden[i] = lstm_out[i, actual_len - 1, :]  # 実際の最終ステップ
            reward = self.reward_predictor(hidden) # 報酬スコア
            continuation_prob = self.continuation_predictor(hidden) # 継続確率

            return reward, continuation_prob
        
        def forward_all_steps(self, state:  torch.Tensor, action: torch.Tensor,
                              lenghts: Optional[torch.Tensor] = None,
                              return_reward: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
            batch_size, max_seq_len, _ = state.shape
            # 状態・行動のエンコード
            state_encoded = self.state_encoder(state.view(-1, state.shape[-1])).view(batch_size, max_seq_len, -1)
            action_encoded = self.action_encoder(action.view(-1, action.shape[-1])).view(batch_size, max_seq_len, -1)
            combined = state_encoded + action_encoded

            ## あんまり理解していない
            lengths_cpu = lengths.cpu()
            sorted_lengths, sorted_idx = lengths_cpu.sort(descending=True)
            _, unsort_idx = sorted_idx.sort()
            
            combined_sorted = combined[sorted_idx]
            packed = nn.utils.rnn.pack_padded_sequence(
                combined_sorted, sorted_lengths, batch_first=True, enforce_sorted=True
            )
            lstm_out_packed, _ = self.lstm(packed)
            lstm_out_sorted, _ = nn.utils.rnn.pad_packed_sequence(
                lstm_out_packed, batch_first=True, total_length=max_seq_len
            )
            lstm_out = lstm_out_sorted[unsort_idx]
            # 各ステップで予測
            lstm_out_flat = lstm_out.reshape(-1, lstm_out.size(-1))
            continuation_flat = self.continuation_predictor(lstm_out_flat).squeeze(-1)
            continuation = continuation_flat.view(batch_size, max_seq_len)

            if return_reward:
                reward_flat = self.reward_predictor(lstm_out_flat).squeeze(-1)
                reward = reward_flat.view(batch_size, max_seq_len)
                return reward, continuation
            return continuation
        
class RetentionIRLRredictor:
    def __init__(self, config: Dict[str, Any]):
        self.state_dim = config.get("state_dim", 10)
        self.action_dim = config.get("action_dim", 5)
        self.hidden_dim = config.get("hidden_dim", 128)
        self.dropout = config.get("dropout", 0.1)
        self.output_temperature = float(config.get("output_temperature", 1.0))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # ネットワークの初期化
        self.network = RetentionIRLNetwork(
            self.state_dim, self.action_dim, self.hidden_dim, self.dropout
        ).to(self.device)

        # オプティマイザの設定
        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=config.get("learning_rate", 0.0003)
        )
        # 損失関数
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()

        self.focal_alpha = 0.25 #クラスの重み
        self.focal_gamma = 1.0  #難易度の重み
        logger.info("継続予測システムを初期化")
    
    def set_focal_loss_params(self, alpha: float, gamma: float):
        self.focal_alpha = alpha
        self.focal_gamma = gamma
        logger.info(f"Focal Lossのパラメータ更新: alpha={alpha}, gamma={gamma}")

    def auto_tune_focal_loss(self, positive_rate: float):
        # ポジティブクラスの割合に応じてパラメータを調整
        if positive_rate >= 0.6:
            alpha = 0.40
            gamma = 1.0
            strategy = "バランス重視（正例率≥60%・軽い正例優先)"
        elif positive_rate >= 0.3:
            alpha = 0.45
            gamma = 1.0
            strategy = "継続重視（正例率30-60%・適度な正例ウェイト)"
        else:
            alpha = 0.55
            gamma = 1.1
            strategy = "継続重視（正例率<30%・正例ウェイト中)"
        
        self.set_focal_loss_params(alpha, gamma)
        logger.info(f"正例率 {positive_rate:.1%} に基づき自動調整: {strategy}")

        def focal_loss(self, predictions: torch.Tensor, targets: torch.Tensor,
                       sample_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
            # Focal Lossの計算
            predictions = predictions.squeeze()
            targets = targets.squeeze()
            # 2値交差エントロピー損失
            bce_loss = F.binary_cross_entropy(predictions, targets, reduction='none')
            # Focal Lossの重み計算
            p_t = predictions * targets + (1 - predictions) * (1 - targets)
            alpha_t = self.focal_alpha * targets + (1 - self.focal_alpha) * (1 - targets)
            focal_weight = alpha_t * torch.pow(1 - p_t, self.focal_gamma)
            focal_loss = focal_weight * bce_loss
        # 依頼なし（拡張期間のみ依頼あり）のサンプルは重みを下げる
        if sample_weights is not None:
            sample_weights = sample_weights.squeeze()
            focal_loss = focal_loss * sample_weights

        # 平均を返す（バッチ全体の損失）
        return focal_loss.mean()
    
    def extract_developer_state(self,
                                developer: Dict[str, Any],
                                activity_history: List[Dict[str, Any]],
                                context_date: datetime) -> DeveloperState:
        # 開発者の状態ベクトルを抽出
        first_seen = developer.get("first_seen", context_date.isoformat())
        if isinstance(first_seen, str):
            first_date = datetime.fromisoformat(first_seen.replace("Z", "+00:00"))
        else:
            first_date = first_seen
        experience_days = (context_date - first_date).days

        total_changes = developer.get('changes_authored', 0)
        total_reviews = developer.get('changes_reviewed', 0)
        # projects = developer.get('projects', [])  # マルチプロジェクト対応時に有効化
        # project_count = len(projects) if isinstance(projects, list) else 0
        # 最近の活動パターン
        recent_activities = self._get_recent_activities(activity_history, context_date, days=30)
        recent_activity_frequency = len(recent_activities) / 30.0
        
        # 活動間隔
        activity_gaps = self._calculate_activity_gaps(activity_history)
        avg_activity_gap = np.mean(activity_gaps) if activity_gaps else 30.0
        
        # 活動トレンド
        activity_trend = self._analyze_activity_trend(activity_history, context_date)
        
        # 協力スコア（簡易版）
        collaboration_score = self._calculate_collaboration_score(activity_history)
        
        # コード品質スコア（簡易版）
        code_quality_score = self._calculate_code_quality_score(activity_history)
        
        # 最近のレビュー受諾率（直近30日）
        recent_acceptance_rate = self._calculate_recent_acceptance_rate(activity_history, context_date, days=30)
        
        # レビュー負荷（直近30日 / 平均）
        review_load = self._calculate_review_load(activity_history, context_date, days=30)
        
        return DeveloperState(
            developer_id=developer.get('developer_id', 'unknown'),
            experience_days=experience_days,
            total_changes=total_changes,
            total_reviews=total_reviews,
            # project_count=project_count,  # マルチプロジェクト対応時に有効化
            recent_activity_frequency=recent_activity_frequency,
            avg_activity_gap=avg_activity_gap,
            activity_trend=activity_trend,
            collaboration_score=collaboration_score,
            code_quality_score=code_quality_score,
            recent_acceptance_rate=recent_acceptance_rate,
            review_load=review_load,
            timestamp=context_date
        )
    
    def extract_developer_action(self,
                                 activity_history: List[Dict[str, Any]],
                                 context_date: datetime) -> DeveloperAction:
        # 開発者の行動ベクトルを抽出
        actions = []
        
        for activity in activity_history:
            try:
                action_type= activity.get('type', 'unknown')
                intensity = self._calulate_action_intensity(activity)
                collaboration = self._calculate_action_collaboration(activity)
                response_time = self._calculate_response_time(activity, context_date)
                review_size = self._calculate_review_size(activity)
                timestamp_str = activity.get('timestamp', context_date.isoformat())
                if isinstance(timestamp_str, str):
                    timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                else:
                    timestamp = timestamp_str
                
                actions.append(DeveloperAction(
                    action_type=action_type,
                    intensity=intensity,
                    collaboration=collaboration,
                    response_time=response_time,
                    review_size=review_size,
                    timestamp=timestamp
                ))
            except Exception as e:
                logger.error(f"Activity processing error: {e}, activity: {activity}")  
                continue

        return actions
    
    def state_to_tensor(self, state: DeveloperState) -> torch.Tensor:
        # 状態をテンソルに変換(0-1の範囲に正規化)

        trend_encoding = {
            'increasing': 1.0,
            'stabele': 0.5,
            'decreasing': 0.0,
            'unknown': 2.5
        }

        # 全特徴量を0-1の範囲に正規化（上限でクリップ）
        features = [
            min(state.experience_days / 730.0, 1.0),  # 2年でキャップ
            min(state.total_changes / 500.0, 1.0),    # 500件でキャップ
            min(state.total_reviews / 500.0, 1.0),    # 500件でキャップ
            # min(state.project_count / 5.0, 1.0),    # マルチプロジェクト対応時に有効化
            min(state.recent_activity_frequency, 1.0), # 既に0-1
            min(state.avg_activity_gap / 60.0, 1.0),  # 60日でキャップ
            trend_encoding.get(state.activity_trend, 0.25), # 既に0-1
            min(state.collaboration_score, 1.0),      # 既に0-1
            min(state.code_quality_score, 1.0),       # 既に0-1
            min(state.recent_acceptance_rate, 1.0),   # 既に0-1（直近30日の受諾率）
            min(state.review_load, 1.0)               # 既に0-1（負荷比率、正規化済み）
        ]

        return torch.tensor(features, dtype=torch.float32, device=self.device)
    
    def action_to_tensor(self, action: DeveloperAction) -> torch.Tensor:
        # 行動をテンソルに変換
        response_speed = 1.0 / (1.0+ action.response_time / 3.0)

        features = [
            min(action.intensity / 10.0, 1.0),
            min(action.collaboration / 10.0, 1.0), 
            response_speed,  # 0-1の範囲
            min(action.review_size / 100.0, 1.0), 
        ]

        return torch.tensor(features, dtype=torch.float32, device=self.device)
    
    def predict_continuation_probability(self,
                                         developer: Dict[str, Any],
                                         activity_history: List[Dict[str, Any]],
                                            context_date: Optional[datetime] = None) -> Dict[str, Any]:
        if context_date is None:
            context_date = datetime.now()

        self.network.eval()

##ここら辺からデバッグ少し雑
        with torch.no_grad():
            actions = self.extract_developer_action(activity_history, context_date)
            if not actions:
                    return {
                        'continuation_probability': 0.5,
                        'confidence': 0.0,
                        'reasoning': '活動履歴が不足しているため、デフォルト確率を返します'
                    }
            # 時系列モード: 全活動を可変長シーケンスとして使用
            # 状態テンソル: 各タイムステップで状態を構築
            state_tensors = []
            states = []  # デバッグ/理由生成用に保存
            for i in range(len(actions)):
                # 各活動時点での状態を抽出
                step_history = activity_history[:i+1]
                step_state = self.extract_developer_state(developer, step_history, context_date)
                states.append(step_state)
                state_tensors.append(self.state_to_tensor(step_state)
                )
            state_seq = torch.stack(state_tensors).unsqueeze(0)  # (1, seq_len, state_dim)

            action_tensors = [self.action_to_tensor(action) for action in actions]
            action_seq = torch.stack(action_tensors).unsqueeze(0)  # (1, seq_len, action_dim)

            lengths = torch.tensor([len(actions)], dtype=torch.long, device=self.device)
            predicted_reward, predicted_continuation = self.network(
                state_seq, action_seq, lengths
            )
            # 最終ステップの状態と行動を取得
            state = states[-1]
            recent_action = actions[-1]

            continuation_prob = predicted_continuation.item()
            reward_score = predicted_reward.item()

            # 信頼度計算（簡易版）
            confidence = min(abs(continuation_prob - 0.5) * 2, 1.0)

            # 理由生成
            reasoning = self._generate_irl_reasoning(
                state, recent_action, continuation_prob, reward_score
            )

            return {
                'continuation_probability': continuation_prob,
                'reward_score': reward_score,
                'confidence': confidence,
                'reasoning': reasoning,
                'state_features': {
                    'experience_days': state.experience_days,
                    'recent_activity_frequency': state.recent_activity_frequency,
                    'collaboration_score': state.collaboration_score,
                    'code_quality_score': state.code_quality_score
                }
            }
