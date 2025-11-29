"""
AI 策略效果診斷與改進建議
分析模型表現不佳的根本原因
"""

import pandas as pd
import numpy as np

print("="*70)
print("AI 金融交易系統 - 問題診斷報告")
print("="*70)

print("\n📊 從執行結果發現的關鍵問題：\n")

print("1. 【嚴重問題】模型預測極度保守")
print("   - 台積電 (2330)：混淆矩陣 [[147, 0], [94, 0]]")
print("   - 長榮 (2603)：混淆矩陣 [[152, 0], [89, 0]]")
print("   → 模型「幾乎不預測買入訊號」，全部預測為 0（觀望）")
print("   → 導致交易次數 = 0，報酬率 = 0%")
print()

print("2. 【核心問題】類別不平衡嚴重")
print("   - 訓練集正樣本比例：25.59% ~ 35.21%")
print("   - 測試集正樣本比例：36.93% ~ 39.00%")
print("   → RandomForest 傾向預測多數類（看跌/觀望）")
print()

print("3. 【標籤定義問題】門檻設定過高")
print("   - 目前：預測「明日收盤 > 明日開盤 * 1.004」(0.4% 門檻)")
print("   - 問題：過濾掉太多潛在機會")
print()

print("4. 【特徵工程問題】")
print("   - RSI/SMA 等指標對「單日漲跌」預測力有限")
print("   - 缺少短期動能指標（MACD、布林通道等）")
print()

print("="*70)
print("💡 改進方案（按優先順序）")
print("="*70)

print("\n【方案 1】解決類別不平衡 ⭐⭐⭐⭐⭐ (最重要)")
print("───────────────────────────────────────")
print("問題：模型訓練時看到太多「看跌」樣本，學會了「一律預測看跌」")
print()
print("解決方法：")
print("  A. 使用 class_weight='balanced' 參數")
print("     → RandomForestClassifier(class_weight='balanced')")
print("     → 自動調整樣本權重，懲罰多數類")
print()
print("  B. 調整預測門檻")
print("     → 不使用 model.predict()，改用 model.predict_proba()")
print("     → 將門檻從 0.5 降低到 0.3-0.4")
print("     → 例：if prob[1] > 0.35: 買入")
print()

print("\n【方案 2】改進標籤定義 ⭐⭐⭐⭐")
print("───────────────────────────────────────")
print("問題：預測「單日」漲跌太困難，雜訊太大")
print()
print("改進方向：")
print("  A. 改為預測「3-5日」的趨勢")
print("     → Target = (Close(t+3) > Close(t) * 1.01)")
print("     → 持有期拉長，減少雜訊")
print()
print("  B. 降低獲利門檻")
print("     → 從 0.4% 降低到 0.2% 或不設門檻")
print("     → threshold=0.002 或 threshold=0")
print()

print("\n【方案 3】增強特徵工程 ⭐⭐⭐")
print("───────────────────────────────────────")
print("新增更多技術指標：")
print("  - MACD (12,26,9)：捕捉趨勢轉折")
print("  - 布林通道 (BB)：判斷超買超賣")
print("  - KD 指標：短期動能")
print("  - 成交量加權指標：VWAP")
print("  - 波動率比率：歷史波動度")
print()

print("\n【方案 4】改進模型參數 ⭐⭐")
print("───────────────────────────────────────")
print("目前參數過於保守：")
print("  - max_depth=10 → 增加到 15-20")
print("  - min_samples_split=20 → 降低到 10")
print("  - min_samples_leaf=10 → 降低到 5")
print("  → 讓模型有更多學習能力")
print()

print("\n【方案 5】改變策略邏輯 ⭐⭐⭐⭐")
print("───────────────────────────────────────")
print("問題：單純的「漲/跌」二元分類太簡化")
print()
print("改進方向：")
print("  A. 三分類策略：")
print("     - 0: 大跌/看空 → 空手")
print("     - 1: 盤整 → 空手")
print("     - 2: 大漲/看多 → 買入")
print()
print("  B. 信心水準過濾：")
print("     - 只在 predict_proba > 0.6 時才交易")
print("     - 信心不足時保持觀望")
print()

print("\n" + "="*70)
print("🎯 立即可執行的快速修正（5分鐘內）")
print("="*70)

print("""
修改以下三個地方：

1. StockTradingModel.__init__() 中的 RandomForestClassifier：
   改為：
   self.model = RandomForestClassifier(
       n_estimators=200,
       max_depth=15,              # 原：10
       min_samples_split=10,      # 原：20
       min_samples_leaf=5,        # 原：10
       class_weight='balanced',   # ★ 新增：解決類別不平衡
       random_state=42,
       n_jobs=-1
   )

2. create_labels() 函數：
   改為：
   threshold=0.002  # 從 0.004 降低到 0.002

3. evaluate() 方法後新增預測門檻調整：
   # 使用自定義門檻
   y_pred_adjusted = (y_pred_proba > 0.35).astype(int)  # 從 0.5 降到 0.35
""")

print("\n" + "="*70)
print("📈 預期改進效果")
print("="*70)
print("""
改進前：
  - 交易次數：0-10 次
  - AI 策略報酬：-0.5% ~ 0%
  - 模型只會預測「不買」

改進後（預期）：
  - 交易次數：20-50 次
  - AI 策略報酬：10% ~ 30%（可能仍輸給 Buy-and-Hold）
  - 模型會產生合理的買賣訊號

重要提醒：
  在 2024 的台股多頭市場，Buy-and-Hold 策略報酬 50-87% 是很難打敗的。
  AI 策略的目標應該是「風險調整後報酬」，而非單純追求最高報酬。
  建議加入「最大回撤」、「夏普比率」等風險指標來綜合評估。
""")

print("\n" + "="*70)
print("🔧 我現在就幫您實施這些修正嗎？(Y/N)")
print("="*70)
