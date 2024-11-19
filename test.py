#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Portfolio Simulation with Dynamic Thresholds and Model Loading with Top 20 Stock Selection
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import joblib
from tqdm import tqdm

# 固定初始参数
INIT_FUND = 1e9
OUTPUT_FILE = "results_dynamic_save.csv"

# 读取数据
test_data_path = "E:/forestregressor/stock_sma_1000.csv"
print("Loading data...")
test_data = pd.read_csv(test_data_path, index_col=0)
test_data = test_data.iloc[47627:59484, :].copy()

# 加载模型
model_path = "E:/forestregressor/trained_rf_model.pkl"  # 替换为你的模型文件路径
rf_model = joblib.load(model_path)
print("模型加载成功")

# 计算 Market Benchmark
average_price = test_data.mean(axis=1)
market_benchmark = (average_price / average_price.iloc[0]) * INIT_FUND

# 初始化交易参数
TRANSACTION_COST = 0.001
BASE_BUY_THRESHOLD = 0.02
BASE_SELL_THRESHOLD = -0.01

cash = INIT_FUND
portfolio = {}
total_assets = []
cash_history = []
portfolio_size_history = []
trade_log = []
volatility_values = []
selected_stock_history = []

# 动态调整阈值函数
def update_thresholds(predicted_returns, base_buy_threshold, base_sell_threshold):
    mean_return = predicted_returns.mean()
    std_return = predicted_returns.std()

    # 动态调整逻辑
    if std_return > 0:  # 确保标准差不为零
        buy_threshold = mean_return + std_return
        sell_threshold = mean_return - std_return / 2
    else:
        buy_threshold = base_buy_threshold
        sell_threshold = base_sell_threshold

    return max(buy_threshold, base_buy_threshold), min(sell_threshold, base_sell_threshold)

# 开始模拟交易
print("Simulating trading...")
for time_step in tqdm(range(1, len(test_data)), desc="Trading Simulation"):
    current_prices = test_data.iloc[time_step]
    past_prices = test_data.iloc[:time_step]

    # 计算特征
    returns = past_prices.pct_change().dropna()
    features = pd.DataFrame({
        "mean_return": returns.mean(),
        "volatility": returns.std(),
        "total_return": (past_prices.iloc[-1] - past_prices.iloc[0]) / past_prices.iloc[0]
    }).reset_index(drop=True)

    # 模型预测
    X_test = features[['mean_return', 'volatility']]
    features['predicted_return'] = rf_model.predict(X_test)

    # 添加股票名称列
    features['Stock'] = test_data.columns

    # 按预测收益排序并选择前20只股票
    features = features.sort_values(by='predicted_return', ascending=False)
    top_stocks = features.head(20)
    selected_stocks = top_stocks['Stock'].tolist()

    # 动态调整阈值
    BUY_THRESHOLD, SELL_THRESHOLD = update_thresholds(features['predicted_return'], BASE_BUY_THRESHOLD, BASE_SELL_THRESHOLD)

    available_cash = cash * 0.9
    for stock in selected_stocks:
        predicted_return = top_stocks.loc[top_stocks['Stock'] == stock, 'predicted_return'].values[0]
        if predicted_return > BUY_THRESHOLD and stock not in portfolio:
            buy_amount = available_cash * 0.05
            shares_to_buy = buy_amount / current_prices[stock]
            portfolio[stock] = shares_to_buy
            cash -= buy_amount * (1 + TRANSACTION_COST)
            trade_log.append({'time_step': time_step, 'action': 'buy', 'stock': stock, 'amount': buy_amount})
        elif predicted_return < SELL_THRESHOLD and stock in portfolio:
            sell_value = portfolio[stock] * current_prices[stock]
            cash += sell_value * (1 - TRANSACTION_COST)
            del portfolio[stock]
            trade_log.append({'time_step': time_step, 'action': 'sell', 'stock': stock, 'amount': sell_value})

    selected_stock_history.append(selected_stocks)

    # 记录总资产和其他指标
    total_value = cash + sum(portfolio.get(stock, 0) * current_prices[stock] for stock in portfolio)
    total_assets.append(total_value)
    cash_history.append(cash)
    portfolio_size_history.append(len(portfolio))
    volatility_values.append(features['volatility'].mean())

# 可视化图形生成
print("Generating visualizations...")
with tqdm(total=1, desc="Plotting Visualizations") as pbar:
    sampled_indices = np.linspace(0, len(total_assets) - 1, num=len(total_assets) // 100, dtype=int)
    sampled_total_assets = [total_assets[i] for i in sampled_indices]
    sampled_market_benchmark = [market_benchmark[i] for i in sampled_indices]

    plt.figure(figsize=(12, 6))
    plt.plot(sampled_total_assets, label="Total Assets (Simulation, Sampled)", color="blue")
    plt.plot(sampled_market_benchmark, label="Market Benchmark (Sampled)", color="orange")
    plt.axhline(y=INIT_FUND, color="gray", linestyle="--", label="Initial Fund")
    plt.title("Total Assets vs Market Benchmark (Sampled Data)")
    plt.xlabel("Time Steps (Sampled)")
    plt.ylabel("Total Assets")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    pbar.update(1)
    
    buy_trades = [t for t in trade_log if t['action'] == 'buy']
    sell_trades = [t for t in trade_log if t['action'] == 'sell']
    
    plt.figure(figsize=(14, 8))
    plt.scatter([t['time_step'] for t in buy_trades], 
                [t['amount'] for t in buy_trades], label='Buy', color='green')
    plt.scatter([t['time_step'] for t in sell_trades], 
                [t['amount'] for t in sell_trades], label='Sell', color='red')
    plt.title("Trading Activity Over Time")
    plt.xlabel("Time Steps (5-Min Intervals)")
    plt.ylabel("Trade Amount")
    plt.legend()
    plt.show()
    pbar.update(1)

# 输出最终结果
final_assets = total_assets[-1]
total_return = (final_assets - INIT_FUND) / INIT_FUND
print(f"Initial Capital: {INIT_FUND:.2f}")
print(f"Final Assets: {final_assets:.2f}")
print(f"Total Return: {total_return * 100:.2f}%")
