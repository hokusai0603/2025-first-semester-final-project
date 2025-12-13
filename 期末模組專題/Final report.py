"""
AI 金融交易系統 - 針對 2020-2024 台股市場
目標：使用機器學習模型預測股票走勢，並與 0050 進行比較

選定標的：
1. 2330.TW 台積電 (權值型/大盤連動高)
2. 2317.TW 鴻海 (AI題材/趨勢型)
3. 2603.TW 長榮 (景氣循環/高波動型)
對標：0050.TW 元大台灣50

"""

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

warnings.filterwarnings('ignore')

# ==================== 第一部分：技術指標計算 ====================

# ==================== 新增：度量學習（原型網路）模型 ====================

class ProtoEncoder(nn.Module):
    def __init__(self, input_dim: int, embedding_dim: int = 16, hidden_dims=(64, 32), dropout: float = 0.1):
        super().__init__()
        layers = []
        last = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(last, h), nn.ReLU()]
            if dropout > 0:
                layers += [nn.Dropout(dropout)]
            last = h
        layers += [nn.Linear(last, embedding_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ProtoNet:
    """簡化版 Prototypical Networks for tabular classification.
    - 使用 MLP 取得 embedding
    - episodic 訓練：每回合對每一類抽 K 支持與 Q 查詢，最小化查詢到各原型距離的交叉熵
    - 推論：以全訓練集嵌入的類別平均向量作為原型，取最近原型
    """

    def __init__(self,
                 input_dim: int,
                 n_classes: int,
                 embedding_dim: int = 16,
                 hidden_dims=(64, 32),
                 dropout: float = 0.1,
                 lr: float = 1e-3,
                 epochs: int = 40,
                 episodes_per_epoch: int = 60,
                 K: int = 5,
                 Q: int = 10,
                 temperature: float = 1.0,
                 device: str = 'cpu'):
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.embedding_dim = embedding_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.episodes_per_epoch = episodes_per_epoch
        self.K = K
        self.Q = Q
        self.temperature = temperature
        self.device = torch.device(device)

        self.encoder = ProtoEncoder(input_dim, embedding_dim, hidden_dims, dropout).to(self.device)
        self.scaler = StandardScaler()
        self.class_to_index = None
        self.index_to_class = None
        self.prototypes = None  # shape: (n_classes, embedding_dim)

    def _to_tensor(self, x_np):
        return torch.tensor(x_np, dtype=torch.float32, device=self.device)

    def _compute_prototypes(self, emb: torch.Tensor, y_idx: np.ndarray) -> torch.Tensor:
        # emb: (N, D), y_idx: (N,)
        protos = []
        for c in range(self.n_classes):
            m = (y_idx == c)
            if m.sum() == 0:
                # 若某類別沒有樣本，設為零向量
                protos.append(torch.zeros(emb.shape[1], device=self.device))
            else:
                protos.append(emb[m].mean(dim=0))
        return torch.stack(protos, dim=0)  # (C, D)

    def fit(self, X: np.ndarray, y: np.ndarray):
        # 建立標籤索引映射
        classes = np.sort(np.unique(y))
        self.class_to_index = {c: i for i, c in enumerate(classes)}
        self.index_to_class = {i: c for c, i in self.class_to_index.items()}
        y_idx = np.vectorize(self.class_to_index.get)(y)

        # 標準化特徵
        X_scaled = self.scaler.fit_transform(X)
        X_tensor = self._to_tensor(X_scaled)

        optimizer = optim.Adam(self.encoder.parameters(), lr=self.lr)

        rng = np.random.default_rng(42)
        # 為每一類建立索引集合
        idx_by_class = {c: np.where(y_idx == c)[0] for c in range(self.n_classes)}

        # 訓練（簡化 episodic）
        self.encoder.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            episodes = 0
            for _ in range(self.episodes_per_epoch):
                support_idx = []
                query_idx = []
                valid = True
                for c in range(self.n_classes):
                    idxs = idx_by_class[c]
                    if len(idxs) == 0:
                        valid = False
                        break
                    # 若數量不足，允許重複抽樣
                    s = rng.choice(idxs, size=self.K, replace=(len(idxs) < self.K))
                    q = rng.choice(idxs, size=self.Q, replace=(len(idxs) < self.Q))
                    support_idx.append(s)
                    query_idx.append(q)
                if not valid:
                    continue

                support_idx = np.concatenate(support_idx)
                query_idx = np.concatenate(query_idx)

                X_sup = X_tensor[support_idx]
                X_que = X_tensor[query_idx]
                y_que = y_idx[query_idx]

                # 取得嵌入與原型
                z_sup = self.encoder(X_sup)  # (C*K, D)
                z_que = self.encoder(X_que)  # (C*Q, D)

                # 計算每類的原型（support 平均）
                # 將 support 拆回各類別切塊
                z_chunks = torch.chunk(z_sup, self.n_classes, dim=0)
                protos = torch.stack([zc.mean(dim=0) for zc in z_chunks], dim=0)  # (C, D)

                # 距離 -> logits
                # z_que: (Nq, D), protos: (C, D)
                # pairwise distances
                dists = torch.cdist(z_que, protos, p=2)  # (Nq, C)
                logits = - (dists ** 2) / self.temperature
                loss = F.cross_entropy(logits, torch.tensor(y_que, device=self.device))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                episodes += 1

            if episodes > 0 and (epoch + 1) % 5 == 0:
                avg_loss = total_loss / episodes
                print(f"ProtoNet 訓練 Epoch {epoch+1}/{self.epochs} - 平均損失: {avg_loss:.4f}")

        # 以所有訓練樣本建立最終原型
        self.encoder.eval()
        with torch.no_grad():
            z_all = self.encoder(X_tensor)
            protos = self._compute_prototypes(z_all, y_idx)
        self.prototypes = protos.detach().cpu().numpy()
        return self

    def _embed(self, X: np.ndarray) -> np.ndarray:
        self.encoder.eval()
        X_scaled = self.scaler.transform(X)
        with torch.no_grad():
            z = self.encoder(self._to_tensor(X_scaled))
        return z.detach().cpu().numpy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        Z = self._embed(X)  # (N, D)
        # 距離到原型
        dists = ((Z[:, None, :] - self.prototypes[None, :, :]) ** 2).sum(axis=2)  # (N, C)
        idx = dists.argmin(axis=1)
        # 映回原始標籤
        return np.vectorize(self.index_to_class.get)(idx)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Z = self._embed(X)
        dists = ((Z[:, None, :] - self.prototypes[None, :, :]) ** 2).sum(axis=2)
        logits = - dists / self.temperature
        # softmax
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        p = e / e.sum(axis=1, keepdims=True)
        # 需對應原始類別順序（0,1,2）
        # self.index_to_class: idx -> class_label
        order = [self.class_to_index[c] for c in sorted(self.class_to_index.keys())]
        return p[:, order]

    # 用於持久化
    def get_state(self):
        return {
            'model_type': 'prototypical',
            'input_dim': self.input_dim,
            'n_classes': self.n_classes,
            'embedding_dim': self.embedding_dim,
            'hidden_dims': self.hidden_dims,
            'dropout': self.dropout,
            'lr': self.lr,
            'epochs': self.epochs,
            'episodes_per_epoch': self.episodes_per_epoch,
            'K': self.K,
            'Q': self.Q,
            'temperature': self.temperature,
            'state_dict': {k: v.cpu().numpy() for k, v in self.encoder.state_dict().items()},
            'prototypes': self.prototypes,
            'scaler_mean_': self.scaler.mean_.copy(),
            'scaler_scale_': self.scaler.scale_.copy(),
            'class_to_index': self.class_to_index,
            'index_to_class': self.index_to_class,
        }

    @staticmethod
    def from_state(state: dict):
        model = ProtoNet(
            input_dim=state['input_dim'],
            n_classes=state['n_classes'],
            embedding_dim=state['embedding_dim'],
            hidden_dims=tuple(state['hidden_dims']),
            dropout=state['dropout'],
            lr=state['lr'],
            epochs=state['epochs'],
            episodes_per_epoch=state['episodes_per_epoch'],
            K=state['K'],
            Q=state['Q'],
            temperature=state['temperature'],
            device='cpu'
        )
        # 還原 encoder 權重
        sd = {k: torch.tensor(v) for k, v in state['state_dict'].items()}
        model.encoder.load_state_dict(sd)
        # 還原 scaler
        model.scaler.mean_ = np.array(state['scaler_mean_'])
        model.scaler.scale_ = np.array(state['scaler_scale_'])
        model.scaler.n_features_in_ = model.input_dim
        # 還原原型與索引映射
        model.prototypes = np.array(state['prototypes'])
        model.class_to_index = state['class_to_index']
        model.index_to_class = state['index_to_class']
        return model

def calculate_rsi(data, period=14):
    """
    計算相對強弱指標 (RSI)
    
    Args:
        data: DataFrame，必須包含 'Close' 欄位
        period: RSI 週期，預設14天
    
    Returns:
        Series: RSI 值 (0-100)
    """
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_sma(data, period=20):
    """
    計算簡單移動平均線 (SMA)
    
    Args:
        data: DataFrame，必須包含 'Close' 欄位
        period: SMA 週期，預設20天
    
    Returns:
        Series: SMA 值
    """
    return data['Close'].rolling(window=period).mean()


def calculate_atr(data, period=14):
    """
    計算真實波動幅度均值 (ATR)
    
    Args:
        data: DataFrame，必須包含 'High', 'Low', 'Close' 欄位
        period: ATR 週期，預設14天
    
    Returns:
        Series: ATR 值
    """
    high_low = data['High'] - data['Low']
    high_close = np.abs(data['High'] - data['Close'].shift())
    low_close = np.abs(data['Low'] - data['Close'].shift())
    
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()
    return atr


def calculate_log_returns(data, periods=[5, 10, 20]):
    """
    計算對數收益率
    
    Args:
        data: DataFrame，必須包含 'Close' 欄位
        periods: 計算週期列表
    
    Returns:
        DataFrame: 包含各週期對數收益率的欄位
    """
    result = pd.DataFrame(index=data.index)
    for period in periods:
        result[f'LogReturn_{period}'] = np.log(data['Close'] / data['Close'].shift(period))
    return result


# ==================== 第二部分：特徵工程 ====================

def engineer_features(data):
    """
    建立完整的技術特徵
    
    Args:
        data: 原始股價 DataFrame
    
    Returns:
        DataFrame: 包含所有技術特徵的 DataFrame
    """
    df = data.copy()
    
    # 1. RSI (標準化至 0-1)
    df['RSI'] = calculate_rsi(df, period=14) / 100.0
    
    # 2. SMA 與乖離率
    df['SMA_20'] = calculate_sma(df, period=20)
    df['SMA_Deviation'] = (df['Close'] - df['SMA_20']) / df['SMA_20']
    
    # 3. ATR (波動率指標)
    df['ATR'] = calculate_atr(df, period=14)
    df['ATR_Normalized'] = df['ATR'] / df['Close']  # 標準化
    
    # 4. 對數收益率
    log_returns = calculate_log_returns(df, periods=[5, 10, 20])
    df = pd.concat([df, log_returns], axis=1)
    
    # 5. 成交量變化率
    df['Volume_Change'] = df['Volume'].pct_change(5)
    
    # 6. 價格動能 (Momentum)
    df['Momentum_10'] = df['Close'] - df['Close'].shift(10)
    df['Momentum_20'] = df['Close'] - df['Close'].shift(20)
    
    # 7. 處理無限值和超大值（替換為 NaN，後續會被 dropna 移除）
    df = df.replace([np.inf, -np.inf], np.nan)
    
    return df


def create_labels(data, threshold=0.004, hold_threshold=0.002):
    """
    建立預測標籤：預測明日的操作策略（三分類）
    考慮交易成本，設定門檻為 0.4% (手續費0.1425% + 證交稅0.3%)
    
    Args:
        data: DataFrame
        threshold: 買入門檻 (預設 0.4%)
        hold_threshold: 持有門檻 (預設 0.2%)
    
    Returns:
        Series: 標籤 (0=賣出/觀望, 1=持有, 2=買入)
    """
    future_close = data['Close'].shift(-1)
    future_open = data['Open'].shift(-1)
    
    # 計算預期收益率
    expected_return = (future_close - future_open) / future_open
    
    # 三分類標籤
    # 2: 預期收益 > threshold (買入)
    # 1: -hold_threshold <= 預期收益 <= threshold (持有)
    # 0: 預期收益 < -hold_threshold (賣出/觀望)
    labels = pd.Series(1, index=data.index)  # 預設為持有
    labels[expected_return > threshold] = 2  # 買入
    labels[expected_return < -hold_threshold] = 0  # 賣出/觀望
    
    return labels


# ==================== 第三部分：資料獲取 ====================

def download_stock_data(stock_id, start_date='2020-01-01', end_date='2024-12-31'):
    """
    下載股票資料
    
    Args:
        stock_id: 股票代碼 (例如 '2330.TW')
        start_date: 開始日期
        end_date: 結束日期
    
    Returns:
        DataFrame: 股價資料
    """
    print(f"正在下載 {stock_id} 的資料...")
    data = yf.download(stock_id, start=start_date, end=end_date, progress=False)
    
    # 處理多層索引問題（當只下載一支股票時，yfinance 可能返回多層索引）
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(1)
    
    print(f"下載完成！共 {len(data)} 筆資料")
    return data


# ==================== 第四部分：模型訓練 ====================

class StockTradingModel:
    """股票交易預測模型類別（已改為度量學習：原型網路）"""
    
    def __init__(self, stock_id, feature_cols=None, use_few_shot=True):
        self.stock_id = stock_id
        self.use_few_shot = use_few_shot  # 是否使用 Few-Shot Learning（透過 episodic 訓練達成）
        self.model = None  # 將在 train() 中建立 ProtoNet
        self.feature_cols = feature_cols or [
            'RSI', 'SMA_Deviation', 'ATR_Normalized',
            'LogReturn_5', 'LogReturn_10', 'LogReturn_20',
            'Volume_Change', 'Momentum_10', 'Momentum_20'
        ]
        self.few_shot_samples = {}
        
    def prepare_data(self, data, train_end='2023-12-31'):
        """
        準備訓練與測試資料
        
        Args:
            data: 完整資料集
            train_end: 訓練集結束日期
        
        Returns:
            tuple: (X_train, X_test, y_train, y_test, train_data, test_data)
        """
        # 建立特徵
        df = engineer_features(data)
        
        # 建立標籤（三分類：0=賣出/觀望, 1=持有, 2=買入）
        df['Target'] = create_labels(df, threshold=0.004, hold_threshold=0.002)
        
        # 移除 NaN
        df = df.dropna()
        
        # 切分訓練集與測試集
        train_data = df.loc[:train_end]
        test_data = df.loc[train_end:]
        
        X_train = train_data[self.feature_cols]
        y_train = train_data['Target']
        X_test = test_data[self.feature_cols]
        y_test = test_data['Target']
        
        print(f"\n訓練集：{len(train_data)} 筆 ({train_data.index[0].date()} ~ {train_data.index[-1].date()})")
        print(f"測試集：{len(test_data)} 筆 ({test_data.index[0].date()} ~ {test_data.index[-1].date()})")
        print(f"訓練集正樣本比例：{y_train.mean():.2%}")
        print(f"測試集正樣本比例：{y_test.mean():.2%}")
        
        return X_train, X_test, y_train, y_test, train_data, test_data
    
    def calculate_sample_weights(self, y_train):
        """
        計算 Few-Shot Learning 的樣本權重
        對於少數類別的樣本給予更高的權重
        
        Args:
            y_train: 訓練標籤
        
        Returns:
            array: 樣本權重
        """
        from sklearn.utils.class_weight import compute_sample_weight
        
        # 計算類別分布
        class_counts = pd.Series(y_train).value_counts()
        print(f"\n類別分布：")
        for class_label in sorted(class_counts.index):
            count = class_counts[class_label]
            percentage = count / len(y_train) * 100
            class_name = ['賣出/觀望', '持有', '買入'][int(class_label)]
            print(f"  {class_name} (類別{int(class_label)}): {count:>4} 筆 ({percentage:>5.2f}%)")
        
        # 使用 sklearn 計算平衡權重
        sample_weights = compute_sample_weight('balanced', y_train)
        
        # Few-Shot 增強：對最少數類別額外增加權重
        min_class = class_counts.idxmin()
        min_count = class_counts.min()
        max_count = class_counts.max()
        
        if min_count < max_count * 0.3:  # 如果最少類別樣本數 < 最多類別的30%
            boost_factor = 1.5  # 額外增強係數
            for i, label in enumerate(y_train):
                if label == min_class:
                    sample_weights[i] *= boost_factor
            print(f"\n套用 Few-Shot Learning：對 '{['賣出/觀望', '持有', '買入'][int(min_class)]}' 類別樣本權重提升 {boost_factor}x")
        
        return sample_weights
    
    def augment_minority_samples(self, X_train, y_train, augment_ratio=0.5):
        """
        數據增強：為少數類別生成合成樣本 (SMOTE-like)
        
        Args:
            X_train: 訓練特徵
            y_train: 訓練標籤
            augment_ratio: 增強比例
        
        Returns:
            tuple: (增強後的X, 增強後的y)
        """
        from sklearn.utils import resample
        
        # 轉換為 numpy array 確保一致性
        if isinstance(X_train, pd.DataFrame):
            X_train_array = X_train.values
        else:
            X_train_array = np.array(X_train)
        
        if isinstance(y_train, pd.Series):
            y_train_array = y_train.values
        else:
            y_train_array = np.array(y_train)
        
        class_counts = pd.Series(y_train_array).value_counts()
        max_count = class_counts.max()
        
        X_augmented = [X_train_array]
        y_augmented = [y_train_array]
        
        # 對每個少數類別進行增強
        for class_label in class_counts.index:
            count = class_counts[class_label]
            if count < max_count * 0.5:  # 如果樣本數 < 最多類別的50%
                # 計算需要增強的樣本數
                n_samples_needed = int((max_count * augment_ratio - count))
                if n_samples_needed > 0:
                    # 從該類別中重新採樣並添加小噪聲
                    class_mask = y_train_array == class_label
                    X_class = X_train_array[class_mask]
                    
                    # 重採樣
                    indices = np.random.choice(len(X_class), n_samples_needed, replace=True)
                    X_resampled = X_class[indices]
                    
                    # 添加小噪聲（5%標準差）
                    noise = np.random.normal(0, 0.05, X_resampled.shape)
                    X_std = np.std(X_resampled, axis=0)
                    X_resampled_noisy = X_resampled + noise * X_std
                    
                    y_resampled = np.array([class_label] * n_samples_needed)
                    
                    X_augmented.append(X_resampled_noisy)
                    y_augmented.append(y_resampled)
                    
                    class_name = ['賣出/觀望', '持有', '買入'][int(class_label)]
                    print(f"  為 '{class_name}' 類別生成 {n_samples_needed} 個增強樣本")
        
        # 合併所有數據
        X_final = np.vstack(X_augmented)
        y_final = np.concatenate(y_augmented)
        
        return X_final, y_final
    
    def train(self, X_train, y_train):
        """訓練模型（使用 Prototypical Network）"""
        print(f"\n開始訓練 {self.stock_id} 的模型...")

        # 印出類別分布（觀察不平衡情況）
        class_counts = pd.Series(y_train).value_counts()
        print("\n類別分布：")
        for class_label in sorted(class_counts.index):
            count = class_counts[class_label]
            percentage = count / len(y_train) * 100
            class_name = ['賣出/觀望', '持有', '買入'][int(class_label)]
            print(f"  {class_name} (類別{int(class_label)}): {count:>4} 筆 ({percentage:>5.2f}%)")

        X_np = X_train.values if isinstance(X_train, pd.DataFrame) else np.asarray(X_train)
        y_np = y_train.values if isinstance(y_train, (pd.Series, pd.DataFrame)) else np.asarray(y_train)

        # 若啟用 few-shot，使用較小 K 並提高 episodes 以強化小樣本學習
        if self.use_few_shot:
            K, Q, episodes = 5, 10, 80
        else:
            K, Q, episodes = 5, 10, 40

        proto = ProtoNet(
            input_dim=X_np.shape[1], n_classes=3,
            embedding_dim=16, hidden_dims=(64, 32), dropout=0.1,
            lr=1e-3, epochs=40, episodes_per_epoch=episodes,
            K=K, Q=Q, temperature=1.0, device='cpu'
        )
        proto.fit(X_np, y_np)
        self.model = proto
        return self.model
    
    def evaluate(self, X_test, y_test):
        """評估模型"""
        X_np = X_test.values if isinstance(X_test, pd.DataFrame) else np.asarray(X_test)
        y_pred = self.model.predict(X_np)
        y_pred_proba = self.model.predict_proba(X_np)
        
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n模型評估結果 - {self.stock_id}")
        print("="*50)
        print(f"準確率 (Accuracy): {accuracy:.4f}")
        print("\n分類報告:")
        print(classification_report(y_test, y_pred, target_names=['賣出/觀望', '持有', '買入']))
        print("\n混淆矩陣:")
        print(confusion_matrix(y_test, y_pred))
        
        return y_pred, y_pred_proba, accuracy
    
    def save_model(self, filepath):
        """儲存模型（支援 Prototypical Network 持久化）"""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.join(script_dir, filepath)

        # 若為 ProtoNet，保存其 state；否則嘗試直接保存（相容舊版）
        if isinstance(self.model, ProtoNet):
            model_state = self.model.get_state()
            model_data = {
                'model_type': 'prototypical',
                'model_state': model_state,
                'feature_cols': self.feature_cols,
                'stock_id': self.stock_id
            }
        else:
            model_data = {
                'model_type': 'legacy',
                'model': self.model,
                'feature_cols': self.feature_cols,
                'stock_id': self.stock_id
            }
        joblib.dump(model_data, full_path)
        print(f"\n模型已儲存至: {full_path}")
    
    @staticmethod
    def load_model(filepath):
        """載入模型（支援 ProtoNet）"""
        if not os.path.isabs(filepath):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            filepath = os.path.join(script_dir, filepath)

        model_data = joblib.load(filepath)
        stock_model = StockTradingModel(
            stock_id=model_data['stock_id'],
            feature_cols=model_data['feature_cols']
        )

        if model_data.get('model_type') == 'prototypical':
            state = model_data['model_state']
            stock_model.model = ProtoNet.from_state(state)
        else:
            stock_model.model = model_data.get('model')

        print(f"模型已載入: {filepath}")
        return stock_model


# ==================== 第五部分：回測系統 ====================

class Backtester:
    """回測系統"""
    
    def __init__(self, initial_capital=1000000, commission_rate=0.002, verbose=True):
        """
        Args:
            initial_capital: 初始資金 (預設100萬)
            commission_rate: 單邊交易成本 (預設0.2%)
        """
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.verbose = verbose
    
    def backtest_portfolio(self, stocks_data, stocks_predictions, allocation_strategy='equal', stocks_pred_proba=None):
        """
        執行多股票投資組合回測
        
        Args:
            stocks_data: dict, {stock_id: test_data}
            stocks_predictions: dict, {stock_id: predictions}
            allocation_strategy: 資金配置策略 ('equal'=平均分配, 'dynamic'=動態配置)
        
        Returns:
            dict: 回測結果統計
        """
        # 確保所有股票的日期索引一致
        all_dates = None
        for stock_id, data in stocks_data.items():
            if all_dates is None:
                all_dates = data.index
            else:
                all_dates = all_dates.intersection(data.index)
        
        # 初始化
        capital = self.initial_capital
        positions = {stock_id: 0 for stock_id in stocks_data.keys()}  # 各股持倉
        trades = []  # 交易記錄
        portfolio_values = []  # 投資組合價值
        
        for i in range(len(all_dates) - 1):
            current_date = all_dates[i]
            
            # 計算當前投資組合價值
            current_value = capital
            for stock_id, position in positions.items():
                if position > 0:
                    current_price = stocks_data[stock_id].loc[current_date, 'Close']
                    current_value += position * current_price * (1 - self.commission_rate)
            
            portfolio_values.append({
                'Date': current_date,
                'Value': current_value
            })
            
            # 獲取下一個交易日的資訊
            next_date = all_dates[i + 1]
            
            # 收集所有股票的訊號
            signals = {}
            for stock_id in stocks_data.keys():
                pred_idx = list(stocks_data[stock_id].index).index(current_date)
                prediction = stocks_predictions[stock_id][pred_idx]
                next_open = stocks_data[stock_id].loc[next_date, 'Open']
                # 取出機率（若提供）
                proba = None
                if stocks_pred_proba and stock_id in stocks_pred_proba:
                    try:
                        proba = stocks_pred_proba[stock_id][pred_idx]
                    except Exception:
                        proba = None
                # 擷取目前的關鍵特徵作為判斷依據（若存在）
                feature_snapshot = {}
                for col in ['RSI', 'SMA_Deviation', 'ATR_Normalized', 'LogReturn_5', 'LogReturn_10', 'LogReturn_20', 'Momentum_10', 'Momentum_20', 'Volume_Change']:
                    if col in stocks_data[stock_id].columns:
                        try:
                            feature_snapshot[col] = float(stocks_data[stock_id].loc[current_date, col])
                        except Exception:
                            pass
                signals[stock_id] = {
                    'prediction': prediction,
                    'next_open': next_open
                    , 'proba': proba
                    , 'features': feature_snapshot
                }
            
            # 決定交易策略
            if allocation_strategy == 'equal':
                # 平均分配策略：將資金平均分配給所有買入訊號的股票
                buy_signals = [sid for sid, sig in signals.items() if sig['prediction'] == 2 and positions[sid] == 0]
                sell_signals = [sid for sid, sig in signals.items() if sig['prediction'] == 0 and positions[sid] > 0]
                
                # 先賣出
                for stock_id in sell_signals:
                    if positions[stock_id] > 0:
                        next_open = signals[stock_id]['next_open']
                        revenue = positions[stock_id] * next_open * (1 - self.commission_rate)
                        capital += revenue
                        # 日誌：輸出判斷依據
                        if self.verbose:
                            msg = f"[SELL] {next_date.date()} {stock_id} 價:{next_open:.2f} 股:{positions[stock_id]} 現金:{capital:,.0f}"
                            p = signals[stock_id].get('proba')
                            if p is not None:
                                msg += f" | 機率(買/持/賣)={p[2]:.2%}/{p[1]:.2%}/{p[0]:.2%}"
                            fs = signals[stock_id].get('features')
                            if fs:
                                # 精簡顯示重點特徵
                                msg += f" | RSI:{fs.get('RSI'):.2f} 乖離:{fs.get('SMA_Deviation'):.2%} ATRn:{fs.get('ATR_Normalized'):.4f}"
                            print(msg)
                        trades.append({
                            'Date': next_date,
                            'Stock': stock_id,
                            'Type': 'SELL',
                            'Price': next_open,
                            'Shares': positions[stock_id],
                            'Capital': capital,
                            'Judgement': {
                                'prediction': 0,
                                'proba': signals[stock_id].get('proba'),
                                'features': signals[stock_id].get('features')
                            }
                        })
                        positions[stock_id] = 0
                
                # 再買入
                if buy_signals:
                    # 將可用資金平均分配
                    capital_per_stock = capital / len(buy_signals)
                    for stock_id in buy_signals:
                        next_open = signals[stock_id]['next_open']
                        shares = int(capital_per_stock / next_open)
                        if shares > 0:
                            cost = shares * next_open * (1 + self.commission_rate)
                            if cost <= capital:
                                capital -= cost
                                positions[stock_id] = shares
                                # 日誌：輸出判斷依據
                                if self.verbose:
                                    msg = f"[BUY ] {next_date.date()} {stock_id} 價:{next_open:.2f} 股:{shares} 現金:{capital:,.0f}"
                                    p = signals[stock_id].get('proba')
                                    if p is not None:
                                        msg += f" | 機率(買/持/賣)={p[2]:.2%}/{p[1]:.2%}/{p[0]:.2%}"
                                    fs = signals[stock_id].get('features')
                                    if fs:
                                        msg += f" | RSI:{fs.get('RSI'):.2f} 乖離:{fs.get('SMA_Deviation'):.2%} ATRn:{fs.get('ATR_Normalized'):.4f}"
                                    print(msg)
                                trades.append({
                                    'Date': next_date,
                                    'Stock': stock_id,
                                    'Type': 'BUY',
                                    'Price': next_open,
                                    'Shares': shares,
                                    'Capital': capital,
                                    'Judgement': {
                                        'prediction': 2,
                                        'proba': signals[stock_id].get('proba'),
                                        'features': signals[stock_id].get('features')
                                    }
                                })
        
        # 最後清算所有持倉
        final_date = all_dates[-1]
        for stock_id, position in positions.items():
            if position > 0:
                final_close = stocks_data[stock_id].loc[final_date, 'Close']
                capital += position * final_close * (1 - self.commission_rate)
                trades.append({
                    'Date': final_date,
                    'Stock': stock_id,
                    'Type': 'SELL',
                    'Price': final_close,
                    'Shares': position,
                    'Capital': capital
                })
        
        # 計算績效指標
        final_value = capital
        total_return = (final_value - self.initial_capital) / self.initial_capital
        
        # 計算最大回撤
        portfolio_df = pd.DataFrame(portfolio_values)
        portfolio_df['Peak'] = portfolio_df['Value'].cummax()
        portfolio_df['Drawdown'] = (portfolio_df['Value'] - portfolio_df['Peak']) / portfolio_df['Peak']
        max_drawdown = portfolio_df['Drawdown'].min()
        
        # 計算年化報酬
        days = (all_dates[-1] - all_dates[0]).days
        years = days / 365.25
        annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
        results = {
            'initial_capital': self.initial_capital,
            'final_value': final_value,
            'total_return': total_return,
            'annual_return': annual_return,
            'max_drawdown': max_drawdown,
            'num_trades': len(trades),
            'trades': trades,
            'portfolio_values': portfolio_df
        }
        
        return results
    
    def print_detailed_trades(self, trades, stocks_names=None):
        """
        詳細印出所有交易記錄，包括交易後的持倉狀態
        
        Args:
            trades: 交易記錄列表
            stocks_names: 股票名稱字典 {stock_id: stock_name}
        """
        if not trades:
            print("\n沒有交易記錄")
            return
        
        print("\n" + "="*120)
        print("詳細交易記錄與持倉狀態")
        print("="*120)
        print(f"{'序號':<5} {'日期':<12} {'股票':<18} {'動作':<8} {'價格':<10} {'股數':<8} {'交易金額':<13} {'剩餘現金':<13} {'當前持倉':<30}")
        print("-"*120)
        
        # 追蹤當前持倉狀態
        current_positions = {}
        
        for idx, trade in enumerate(trades, 1):
            date_str = trade['Date'].strftime('%Y-%m-%d')
            stock_id = trade['Stock']
            stock_name = stocks_names.get(stock_id, stock_id) if stocks_names else stock_id
            trade_type = trade['Type']
            price = trade['Price']
            shares = trade['Shares']
            capital = trade['Capital']
            
            # 計算交易金額
            if trade_type == 'BUY':
                amount = shares * price * (1 + self.commission_rate)
                action_symbol = '買入 ↑'
                current_positions[stock_id] = current_positions.get(stock_id, 0) + shares
            else:
                amount = shares * price * (1 - self.commission_rate)
                action_symbol = '賣出 ↓'
                current_positions[stock_id] = current_positions.get(stock_id, 0) - shares
                if current_positions[stock_id] == 0:
                    del current_positions[stock_id]
            
            # 顯示股票名稱
            display_name = f"{stock_name[:6]}({stock_id[:7]})"
            
            # 顯示當前持倉
            if current_positions:
                positions_str = ", ".join([f"{sid[:7]}:{shares}股" for sid, shares in current_positions.items()])
            else:
                positions_str = "空倉"
            
            base = f"{idx:<5} {date_str:<12} {display_name:<18} {action_symbol:<8} ${price:>8.2f} {shares:>7} ${amount:>11,.0f} ${capital:>11,.0f} {positions_str:<30}"
            # 若有判斷依據，精簡附註
            jud = trade.get('Judgement')
            if jud and self.verbose:
                p = jud.get('proba')
                fs = jud.get('features') or {}
                note = []
                if p is not None:
                    note.append(f"機率(買/持/賣)={p[2]:.2%}/{p[1]:.2%}/{p[0]:.2%}")
                # 只擷取重點特徵
                if fs:
                    try:
                        note.append(f"RSI:{float(fs.get('RSI')):.2f}")
                    except Exception:
                        pass
                    try:
                        note.append(f"乖離:{float(fs.get('SMA_Deviation')):.2%}")
                    except Exception:
                        pass
                    try:
                        note.append(f"ATRn:{float(fs.get('ATR_Normalized')):.4f}")
                    except Exception:
                        pass
                if note:
                    base += " \n      (" + "; ".join(note) + ")"
            print(base)
        
        print("="*120)
        print(f"總計 {len(trades)} 筆交易")
        print("="*120)
        
    def backtest(self, test_data, predictions, predictions_proba=None):
        """
        執行回測
        
        Args:
            test_data: 測試集資料
            predictions: 模型預測結果
        
        Returns:
            dict: 回測結果統計
        """
        df = test_data.copy()
        df['Prediction'] = predictions
        
        # 初始化
        capital = self.initial_capital
        position = 0  # 持倉數量
        trades = []  # 交易記錄
        portfolio_values = []  # 投資組合價值
        
        for i in range(len(df) - 1):
            current_date = df.index[i]
            next_open = df.iloc[i + 1]['Open']
            current_prediction = df.iloc[i]['Prediction']
            current_proba = None
            if predictions_proba is not None:
                try:
                    current_proba = predictions_proba[i]
                except Exception:
                    current_proba = None
            
            # 根據預測訊號決定交易（0=賣出, 1=持有, 2=買入）
            if current_prediction == 2 and position == 0:
                # 買入訊號：用所有資金買入
                shares = int(capital / next_open)
                if shares > 0:
                    cost = shares * next_open * (1 + self.commission_rate)
                    capital -= cost
                    position = shares
                    # 輸出判斷依據日誌
                    if self.verbose:
                        msg = f"[BUY ] {df.index[i + 1].date()} 價:{next_open:.2f} 股:{shares} 現金:{capital:,.0f}"
                        if current_proba is not None:
                            msg += f" | 機率(買/持/賣)={current_proba[2]:.2%}/{current_proba[1]:.2%}/{current_proba[0]:.2%}"
                        # 精簡特徵
                        fs = {k: float(df.iloc[i].get(k)) for k in ['RSI','SMA_Deviation','ATR_Normalized'] if k in df.columns}
                        if fs:
                            msg += f" | RSI:{fs.get('RSI'):.2f} 乖離:{fs.get('SMA_Deviation'):.2%} ATRn:{fs.get('ATR_Normalized'):.4f}"
                        print(msg)
                    trades.append({
                        'Date': df.index[i + 1],
                        'Type': 'BUY',
                        'Price': next_open,
                        'Shares': shares,
                        'Capital': capital,
                        'Judgement': {
                            'prediction': 2,
                            'proba': current_proba,
                            'features': {k: float(df.iloc[i].get(k)) for k in ['RSI','SMA_Deviation','ATR_Normalized','LogReturn_5','LogReturn_10','LogReturn_20','Momentum_10','Momentum_20','Volume_Change'] if k in df.columns}
                        }
                    })
            
            elif current_prediction == 0 and position > 0:
                # 賣出訊號：清倉
                revenue = position * next_open * (1 - self.commission_rate)
                capital += revenue
                if self.verbose:
                    msg = f"[SELL] {df.index[i + 1].date()} 價:{next_open:.2f} 股:{position} 現金:{capital:,.0f}"
                    if current_proba is not None:
                        msg += f" | 機率(買/持/賣)={current_proba[2]:.2%}/{current_proba[1]:.2%}/{current_proba[0]:.2%}"
                    fs = {k: float(df.iloc[i].get(k)) for k in ['RSI','SMA_Deviation','ATR_Normalized'] if k in df.columns}
                    if fs:
                        msg += f" | RSI:{fs.get('RSI'):.2f} 乖離:{fs.get('SMA_Deviation'):.2%} ATRn:{fs.get('ATR_Normalized'):.4f}"
                    print(msg)
                trades.append({
                    'Date': df.index[i + 1],
                    'Type': 'SELL',
                    'Price': next_open,
                    'Shares': position,
                    'Capital': capital,
                    'Judgement': {
                        'prediction': 0,
                        'proba': current_proba,
                        'features': {k: float(df.iloc[i].get(k)) for k in ['RSI','SMA_Deviation','ATR_Normalized','LogReturn_5','LogReturn_10','LogReturn_20','Momentum_10','Momentum_20','Volume_Change'] if k in df.columns}
                    }
                })
                position = 0
            
            # 當 current_prediction == 1 時，保持持有狀態（不做任何動作）
            
            # 計算當前投資組合價值
            current_value = capital
            if position > 0:
                current_value += position * df.iloc[i]['Close'] * (1 - self.commission_rate)
            
            portfolio_values.append({
                'Date': current_date,
                'Value': current_value
            })
        
        # 最後如果還有持倉，按最後一天收盤價賣出
        if position > 0:
            final_close = df.iloc[-1]['Close']
            capital += position * final_close * (1 - self.commission_rate)
            trades.append({
                'Date': df.index[-1],
                'Type': 'SELL',
                'Price': final_close,
                'Shares': position,
                'Capital': capital
            })
        
        # 計算績效指標
        final_value = capital
        total_return = (final_value - self.initial_capital) / self.initial_capital
        
        # 計算最大回撤
        portfolio_df = pd.DataFrame(portfolio_values)
        portfolio_df['Peak'] = portfolio_df['Value'].cummax()
        portfolio_df['Drawdown'] = (portfolio_df['Value'] - portfolio_df['Peak']) / portfolio_df['Peak']
        max_drawdown = portfolio_df['Drawdown'].min()
        
        # 計算年化報酬
        days = (df.index[-1] - df.index[0]).days
        years = days / 365.25
        annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
        results = {
            'initial_capital': self.initial_capital,
            'final_value': final_value,
            'total_return': total_return,
            'annual_return': annual_return,
            'max_drawdown': max_drawdown,
            'num_trades': len(trades),
            'trades': trades,
            'portfolio_values': portfolio_df
        }
        
        return results
    
    def calculate_buy_and_hold(self, test_data):
        """計算買入持有策略的績效"""
        first_price = test_data.iloc[0]['Open']
        last_price = test_data.iloc[-1]['Close']
        
        shares = int(self.initial_capital / first_price)
        buy_cost = shares * first_price * (1 + self.commission_rate)
        sell_revenue = shares * last_price * (1 - self.commission_rate)
        
        final_value = sell_revenue + (self.initial_capital - buy_cost)
        total_return = (final_value - self.initial_capital) / self.initial_capital
        
        days = (test_data.index[-1] - test_data.index[0]).days
        years = days / 365.25
        annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
        return {
            'final_value': final_value,
            'total_return': total_return,
            'annual_return': annual_return
        }
    
    def print_results(self, results, benchmark_results=None):
        """印出回測結果"""
        print("\n" + "="*70)
        print("回測結果摘要")
        print("="*70)
        print(f"初始資金：${results['initial_capital']:,.0f}")
        print(f"最終資金：${results['final_value']:,.0f}")
        print(f"總報酬率：{results['total_return']:.2%}")
        print(f"年化報酬：{results['annual_return']:.2%}")
        print(f"最大回撤：{results['max_drawdown']:.2%}")
        print(f"交易次數：{results['num_trades']}")
        
        if benchmark_results:
            print("\n" + "-"*70)
            print("基準比較 (Buy-and-Hold)")
            print("-"*70)
            print(f"基準總報酬率：{benchmark_results['total_return']:.2%}")
            print(f"基準年化報酬：{benchmark_results['annual_return']:.2%}")
            print(f"超額報酬：{(results['total_return'] - benchmark_results['total_return']):.2%}")
        
        print("="*70)


# ==================== 第六部分：每日互動推論 ====================

def daily_inference(stock_id, model_path):
    """
    每日收盤後執行的推論腳本
    
    Args:
        stock_id: 股票代碼
        model_path: 模型路徑
    
    Returns:
        dict: 推論結果
    """
    print(f"\n{'='*70}")
    print(f"每日推論 - {stock_id}")
    print(f"執行時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")
    
    # 1. 載入模型
    model = StockTradingModel.load_model(model_path)
    
    # 2. 下載最新資料 (需要足夠的歷史資料以計算技術指標)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=120)  # 抓取近4個月資料
    
    data = yf.download(stock_id, start=start_date.strftime('%Y-%m-%d'), 
                      end=end_date.strftime('%Y-%m-%d'), progress=False)
    
    if len(data) == 0:
        print("錯誤：無法獲取資料")
        return None
    
    # 3. 建立特徵
    df = engineer_features(data)
    df = df.dropna()
    
    if len(df) == 0:
        print("錯誤：特徵計算後無有效資料")
        return None
    
    # 4. 取最新一筆資料進行預測
    latest_features = df[model.feature_cols].iloc[[-1]]
    latest_date = df.index[-1]
    latest_close = df.iloc[-1]['Close']
    
    # 5. 模型推論
    prediction = model.model.predict(latest_features)[0]
    probabilities = model.model.predict_proba(latest_features)[0]
    
    # 6. 輸出結果
    action_map = {0: '賣出/觀望', 1: '持有', 2: '買入'}
    print(f"\n最新資料日期：{latest_date.date()}")
    print(f"收盤價：${latest_close:.2f}")
    print(f"\n預測結果：{action_map[prediction]} (建議操作)")
    print(f"買入機率：{probabilities[2]:.2%}")
    print(f"持有機率：{probabilities[1]:.2%}")
    print(f"賣出/觀望機率：{probabilities[0]:.2%}")
    
    # 7. 顯示關鍵技術指標
    print(f"\n關鍵技術指標：")
    print(f"  RSI：{df.iloc[-1]['RSI']*100:.2f}")
    print(f"  SMA乖離率：{df.iloc[-1]['SMA_Deviation']:.2%}")
    print(f"  ATR標準化：{df.iloc[-1]['ATR_Normalized']:.4f}")
    print(f"  5日對數收益：{df.iloc[-1]['LogReturn_5']:.4f}")
    
    signal_map = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
    result = {
        'stock_id': stock_id,
        'date': latest_date,
        'close_price': latest_close,
        'prediction': prediction,
        'probabilities': probabilities,
        'signal': signal_map[prediction]
    }
    
    print(f"\n{'='*70}\n")
    
    return result


# ==================== 第七部分：主程式 ====================

def main():
    """主程式：執行完整的訓練與回測流程"""
    
    print("="*70)
    print("AI 金融交易系統 - 台股 MVP 專案 (投資組合模式)")
    print("="*70)
    
    # 定義標的股票
    stocks = {
        '2330.TW': '台積電 (權值型)',
        '2317.TW': '鴻海 (趨勢型)',
        '2603.TW': '長榮 (高波動型)'
    }
    
    benchmark = '0050.TW'  # 基準指數
    
    # 儲存所有模型和預測結果
    all_models = {}
    all_test_data = {}
    all_predictions = {}
    all_predictions_proba = {}
    all_accuracies = {}
    
    # 取得腳本所在目錄
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 第一階段：訓練/載入所有股票的模型
    print("\n" + "="*70)
    print("第一階段：訓練/載入模型")
    print("="*70)
    
    for stock_id, stock_name in stocks.items():
        print(f"\n\n{'#'*70}")
        print(f"處理股票：{stock_name} ({stock_id})")
        print(f"{'#'*70}")
        
        try:
            # 定義模型檔案路徑
            model_filename = f'model_{stock_id.replace(".TW", "")}.pkl'
            model_path = os.path.join(script_dir, model_filename)
            
            # 檢查模型是否已存在
            if os.path.exists(model_path):
                print(f"\n偵測到已存在的模型檔案: {model_path}")
                print("正在載入模型...")
                model = StockTradingModel.load_model(model_filename)
                
                # 下載資料用於回測
                data = download_stock_data(stock_id)
                df = engineer_features(data)
                df['Target'] = create_labels(df, threshold=0.004, hold_threshold=0.002)
                df = df.dropna()
                test_data = df.loc['2024-01-01':]
                
                # 使用載入的模型進行預測
                X_test = test_data[model.feature_cols]
                y_test = test_data['Target']
                y_pred, y_pred_proba, accuracy = model.evaluate(X_test, y_test)
                
            else:
                print(f"\n未找到模型檔案，開始訓練新模型...")
                
                # 1. 下載資料
                data = download_stock_data(stock_id)
                
                # 2. 初始化模型
                model = StockTradingModel(stock_id)
                
                # 3. 準備資料
                X_train, X_test, y_train, y_test, train_data, test_data = model.prepare_data(data)
                
                # 4. 訓練模型
                model.train(X_train, y_train)
                
                # 5. 評估模型
                y_pred, y_pred_proba, accuracy = model.evaluate(X_test, y_test)
                
                # 6. 儲存模型
                model.save_model(model_filename)
            
            # 儲存結果
            all_models[stock_id] = model
            all_test_data[stock_id] = test_data
            all_predictions[stock_id] = y_pred
            all_accuracies[stock_id] = accuracy
            all_predictions_proba[stock_id] = y_pred_proba
            
        except Exception as e:
            print(f"錯誤：處理 {stock_id} 時發生異常：{str(e)}")
            continue
    
    # 第二階段：使用投資組合策略回測（共用100萬資金）
    print("\n\n" + "="*70)
    print("第二階段：投資組合回測（三檔股票共用100萬資金）")
    print("="*70)
    
    if len(all_test_data) == len(stocks):
        backtester = Backtester(initial_capital=1000000, verbose=True)
        portfolio_results = backtester.backtest_portfolio(all_test_data, all_predictions, stocks_pred_proba=all_predictions_proba)
        
        print("\n投資組合回測結果：")
        print("="*70)
        print(f"初始資金：${portfolio_results['initial_capital']:,.0f}")
        print(f"最終資金：${portfolio_results['final_value']:,.0f}")
        print(f"總報酬率：{portfolio_results['total_return']:.2%}")
        print(f"年化報酬：{portfolio_results['annual_return']:.2%}")
        print(f"最大回撤：{portfolio_results['max_drawdown']:.2%}")
        print(f"總交易次數：{portfolio_results['num_trades']}")
        
        # 統計各股票的交易次數
        trade_counts = {}
        for trade in portfolio_results['trades']:
            stock = trade['Stock']
            trade_counts[stock] = trade_counts.get(stock, 0) + 1
        
        print("\n各股票交易次數：")
        for stock_id, count in trade_counts.items():
            stock_name = stocks[stock_id]
            print(f"  {stock_name} ({stock_id}): {count} 次")
        
        print("="*70)
        
        # 顯示詳細交易記錄
        if portfolio_results['num_trades'] > 0:
            backtester.print_detailed_trades(portfolio_results['trades'], stocks)
    else:
        print("\n警告：部分股票模型未能成功載入，無法執行投資組合回測")
        portfolio_results = None
    
    # 第三階段：下載並比較 0050
    print(f"\n\n{'#'*70}")
    print(f"比較基準指數：0050.TW")
    print(f"{'#'*70}")
    
    try:
        benchmark_data = download_stock_data(benchmark)
        test_benchmark = benchmark_data.loc['2024-01-01':]
        
        backtester_bench = Backtester(initial_capital=1000000)
        benchmark_0050 = backtester_bench.calculate_buy_and_hold(test_benchmark)
        
        print("\n0050 買入持有策略績效：")
        total_return = float(benchmark_0050['total_return'])
        annual_return = float(benchmark_0050['annual_return'])
        print(f"總報酬率：{total_return:.2%}")
        print(f"年化報酬：{annual_return:.2%}")
        
    except Exception as e:
        print(f"錯誤：處理 0050 時發生異常：{str(e)}")
        benchmark_0050 = None
    
    # 最終總結
    print("\n\n" + "="*70)
    print("最終績效總結")
    print("="*70)
    
    print("\n【各股票模型準確率】")
    print("-"*70)
    for stock_id, accuracy in all_accuracies.items():
        stock_name = stocks[stock_id]
        print(f"{stock_name:<20} ({stock_id}): {accuracy:.2%}")
    
    if portfolio_results:
        print("\n【投資組合策略 vs 基準指數】")
        print("-"*70)
        print(f"{'策略':<30} {'總報酬率':<15} {'年化報酬':<15} {'最大回撤':<15}")
        print("-"*70)
        print(f"{'AI投資組合 (三檔共100萬)':<30} {portfolio_results['total_return']:>12.2%}  {portfolio_results['annual_return']:>12.2%}  {portfolio_results['max_drawdown']:>12.2%}")
        
        if benchmark_0050:
            print(f"{'0050買入持有 (100萬)':<30} {benchmark_0050['total_return']:>12.2%}  {benchmark_0050['annual_return']:>12.2%}  {'N/A':>12}")
            excess = portfolio_results['total_return'] - benchmark_0050['total_return']
            print("-"*70)
            print(f"{'超額報酬':<30} {excess:>12.2%}")
    
    print("="*70)
    
    print("\n專案執行完成！")
    print("投資組合模式：使用100萬資金同時操作三檔股票")
    print("您可以使用 daily_inference() 函數進行每日推論")
    print("範例：daily_inference('2330.TW', 'model_2330.pkl')")
    
    return {
        'portfolio_results': portfolio_results,
        'accuracies': all_accuracies,
        'benchmark_0050': benchmark_0050
    }


if __name__ == "__main__":
    # 執行主程式
    results = main()
    
    # 可選：執行每日推論範例
    # daily_inference('2330.TW', 'model_2330.pkl')
