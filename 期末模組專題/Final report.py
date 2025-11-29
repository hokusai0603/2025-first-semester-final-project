"""
AI 金融交易系統 - 針對 2022-2024 台股市場
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import warnings
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os

warnings.filterwarnings('ignore')

# ==================== 第一部分：技術指標計算 ====================

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


def create_labels(data, threshold=0.002):
    """
    建立預測標籤：預測明日收盤價是否高於明日開盤價
    考慮交易成本，設定門檻為 0.2% (降低門檻以增加機會)
    
    Args:
        data: DataFrame
        threshold: 獲利門檻 (預設 0.2%)
    
    Returns:
        Series: 標籤 (1=看漲買入, 0=看跌觀望)
    """
    future_close = data['Close'].shift(-1)
    future_open = data['Open'].shift(-1)
    
    # 只有當預期收益超過交易成本時才標記為 1
    labels = (future_close > future_open * (1 + threshold)).astype(int)
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
    """股票交易預測模型類別"""
    
    def __init__(self, stock_id, feature_cols=None):
        self.stock_id = stock_id
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,              # 原：10 → 增加深度
            min_samples_split=10,      # 原：20 → 降低限制
            min_samples_leaf=5,        # 原：10 → 降低限制
            class_weight='balanced',   # ★ 新增：解決類別不平衡
            random_state=42,
            n_jobs=-1
        )
        self.feature_cols = feature_cols or [
            'RSI', 'SMA_Deviation', 'ATR_Normalized',
            'LogReturn_5', 'LogReturn_10', 'LogReturn_20',
            'Volume_Change', 'Momentum_10', 'Momentum_20'
        ]
        self.scaler = None
        
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
        
        # 建立標籤 (使用降低後的門檻)
        df['Target'] = create_labels(df, threshold=0.002)
        
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
    
    def train(self, X_train, y_train):
        """訓練模型"""
        print(f"\n開始訓練 {self.stock_id} 的模型...")
        self.model.fit(X_train, y_train)
        
        # 顯示特徵重要性
        feature_importance = pd.DataFrame({
            'Feature': self.feature_cols,
            'Importance': self.model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print("\n特徵重要性 (前5名):")
        print(feature_importance.head())
        
        return self.model
    
    def evaluate(self, X_test, y_test, threshold=0.35):
        """
        評估模型
        
        Args:
            X_test: 測試特徵
            y_test: 測試標籤
            threshold: 預測機率門檻 (預設 0.35，降低門檻以增加買入訊號)
        
        Returns:
            tuple: (y_pred, y_pred_proba, accuracy)
        """
        # 使用自定義門檻進行預測
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= threshold).astype(int)  # ★ 使用自定義門檻
        
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n模型評估結果 - {self.stock_id}")
        print("="*50)
        print(f"準確率 (Accuracy): {accuracy:.4f}")
        print(f"預測門檻 (Threshold): {threshold}")
        print(f"預測買入次數: {y_pred.sum()} / {len(y_pred)} ({y_pred.mean():.2%})")
        print("\n分類報告:")
        print(classification_report(y_test, y_pred, target_names=['看跌/觀望', '看漲/買入']))
        print("\n混淆矩陣:")
        print(confusion_matrix(y_test, y_pred))
        
        return y_pred, y_pred_proba, accuracy
    
    def save_model(self, filepath):
        """儲存模型"""
        # 確保使用腳本所在目錄的絕對路徑
        script_dir = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.join(script_dir, filepath)
        
        model_data = {
            'model': self.model,
            'feature_cols': self.feature_cols,
            'stock_id': self.stock_id
        }
        joblib.dump(model_data, full_path)
        print(f"\n模型已儲存至: {full_path}")
    
    @staticmethod
    def load_model(filepath):
        """載入模型"""
        # 如果是相對路徑，轉換為腳本目錄的絕對路徑
        if not os.path.isabs(filepath):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            filepath = os.path.join(script_dir, filepath)
        
        model_data = joblib.load(filepath)
        stock_model = StockTradingModel(
            stock_id=model_data['stock_id'],
            feature_cols=model_data['feature_cols']
        )
        stock_model.model = model_data['model']
        print(f"模型已載入: {filepath}")
        return stock_model


# ==================== 第五部分：回測系統 ====================

class Backtester:
    """回測系統"""
    
    def __init__(self, initial_capital=1000000, commission_rate=0.002):
        """
        Args:
            initial_capital: 初始資金 (預設100萬)
            commission_rate: 單邊交易成本 (預設0.2%)
        """
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        
    def backtest(self, test_data, predictions):
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
            
            # 根據預測訊號決定交易
            if current_prediction == 1 and position == 0:
                # 買入訊號：用所有資金買入
                shares = int(capital / next_open)
                if shares > 0:
                    cost = shares * next_open * (1 + self.commission_rate)
                    capital -= cost
                    position = shares
                    trades.append({
                        'Date': df.index[i + 1],
                        'Type': 'BUY',
                        'Price': next_open,
                        'Shares': shares,
                        'Capital': capital
                    })
            
            elif current_prediction == 0 and position > 0:
                # 賣出訊號：清倉
                revenue = position * next_open * (1 - self.commission_rate)
                capital += revenue
                trades.append({
                    'Date': df.index[i + 1],
                    'Type': 'SELL',
                    'Price': next_open,
                    'Shares': position,
                    'Capital': capital
                })
                position = 0
            
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
    print(f"\n最新資料日期：{latest_date.date()}")
    print(f"收盤價：${latest_close:.2f}")
    print(f"\n預測結果：{'看漲 (建議買入)' if prediction == 1 else '看跌/盤整 (建議觀望)'}")
    print(f"看漲機率：{probabilities[1]:.2%}")
    print(f"看跌機率：{probabilities[0]:.2%}")
    
    # 7. 顯示關鍵技術指標
    print(f"\n關鍵技術指標：")
    print(f"  RSI：{df.iloc[-1]['RSI']*100:.2f}")
    print(f"  SMA乖離率：{df.iloc[-1]['SMA_Deviation']:.2%}")
    print(f"  ATR標準化：{df.iloc[-1]['ATR_Normalized']:.4f}")
    print(f"  5日對數收益：{df.iloc[-1]['LogReturn_5']:.4f}")
    
    result = {
        'stock_id': stock_id,
        'date': latest_date,
        'close_price': latest_close,
        'prediction': prediction,
        'probabilities': probabilities,
        'signal': 'BUY' if prediction == 1 else 'HOLD'
    }
    
    print(f"\n{'='*70}\n")
    
    return result


# ==================== 第七部分：主程式 ====================

def main():
    """主程式：執行完整的訓練與回測流程"""
    
    print("="*70)
    print("AI 金融交易系統 - 台股 MVP 專案")
    print("="*70)
    
    # 定義標的股票
    stocks = {
        '2330.TW': '台積電 (權值型)',
        '2317.TW': '鴻海 (趨勢型)',
        '2603.TW': '長榮 (高波動型)'
    }
    
    benchmark = '0050.TW'  # 基準指數
    
    # 儲存所有結果
    all_results = {}
    
    # 取得腳本所在目錄
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 對每支股票執行訓練與回測
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
                df['Target'] = create_labels(df, threshold=0.002)  # 使用新門檻
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
            
            # 7. 回測
            backtester = Backtester(initial_capital=1000000)
            backtest_results = backtester.backtest(test_data, y_pred)
            buy_hold_results = backtester.calculate_buy_and_hold(test_data)
            
            # 8. 顯示結果
            backtester.print_results(backtest_results, buy_hold_results)
            
            # 儲存結果
            all_results[stock_id] = {
                'stock_name': stock_name,
                'accuracy': accuracy,
                'backtest': backtest_results,
                'buy_hold': buy_hold_results,
                'model_path': model_path
            }
            
        except Exception as e:
            print(f"錯誤：處理 {stock_id} 時發生異常：{str(e)}")
            continue
    
    # 下載並比較 0050
    print(f"\n\n{'#'*70}")
    print(f"下載基準指數：0050.TW")
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
    
    # 總結比較
    print("\n\n" + "="*70)
    print("總結：各股票策略績效比較")
    print("="*70)
    print(f"{'股票代碼':<12} {'股票名稱':<15} {'模型準確率':<12} {'AI策略報酬':<12} {'買入持有報酬':<12} {'超額報酬':<10}")
    print("-"*70)
    
    for stock_id, results in all_results.items():
        stock_name = results['stock_name']
        accuracy = results['accuracy']
        ai_return = results['backtest']['total_return']
        bh_return = results['buy_hold']['total_return']
        excess_return = ai_return - bh_return
        
        print(f"{stock_id:<12} {stock_name:<15} {accuracy:>10.2%}  {ai_return:>10.2%}  {bh_return:>10.2%}  {excess_return:>10.2%}")
    
    if benchmark_0050:
        print("-"*70)
        print(f"{'0050.TW':<12} {'元大台灣50':<15} {'N/A':>10}  {'N/A':>10}  {benchmark_0050['total_return']:>10.2%}  {'N/A':>10}")
    
    print("="*70)
    
    print("\n專案執行完成！")
    print("您可以使用 daily_inference() 函數進行每日推論")
    print("範例：daily_inference('2330.TW', 'model_2330.pkl')")
    
    return all_results


if __name__ == "__main__":
    # 執行主程式
    results = main()
    
    # 可選：執行每日推論範例
    # daily_inference('2330.TW', 'model_2330.pkl')
