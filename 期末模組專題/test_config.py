"""測試配置和決策解釋功能"""
import sys
sys.path.append(r"c:\Users\lcc04\Desktop\homework for an AI project\期末報告喔~\2025-first-semester-final-project\期末模組專題")

# 測試導入
try:
    from Final_report import TradingConfig
    print("✅ TradingConfig 導入成功")
    
    config = TradingConfig()
    print(f"\n📊 當前配置：")
    print(f"   Embedding維度: {config.EMBEDDING_DIM}")
    print(f"   隱藏層: {config.HIDDEN_DIMS}")
    print(f"   Dropout: {config.DROPOUT}")
    print(f"   學習率: {config.LEARNING_RATE}")
    print(f"   訓練輪數: {config.EPOCHS}")
    print(f"   買入門檻: {config.BUY_THRESHOLD:.2%}")
    print(f"   賣出門檻: {config.SELL_THRESHOLD:.2%}")
    print("\n✅ 所有配置參數讀取正常")
    
except ImportError as e:
    print(f"❌ 導入錯誤: {e}")
except Exception as e:
    print(f"❌ 其他錯誤: {e}")
