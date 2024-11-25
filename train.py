#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Dynamic Portfolio Management with Adaptive Thresholds and Model Saving
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from tqdm import tqdm  # 导入进度条模块
import joblib  # 用于保存和加载模型
import seaborn as sns
from collections import Counter

# 初始参数
INIT_FUND = 1e9
BASE_BUY_THRESHOLD = 0.01
BASE_SELL_THRESHOLD = -0.01
TRANSACTION_COST = 0.001
BASE_TRAIN_INTERVAL = 1000  # 默认训练间隔
N_ESTIMATORS = 100  # 优化模型训练时间
MAX_DEPTH = 20  # 控制树的深度
WINDOW_SIZE = 100  # 动态调整滑动窗口大小

# 读取数据
print("Loading data...")
stock_data = pd.read_csv("E:/forestregressor/stock_sma_1000.csv", index_col=0)

# 裁剪数据，确保格式正确
stock_data = stock_data.iloc[1000:, :].copy()
assert stock_data.shape == (58483, 100), "数据维度与预期不符，请检查数据格式！"

# 初始化资金、持仓和日志
cash = INIT_FUND
portfolio = {}
total_assets = []
trade_log = []
selected_stocks = []  # 存储动态选出的股票
selected_stock_history = []  # 记录每次选中的股票
rf_model = None  # 初始化模型
train_interval = BASE_TRAIN_INTERVAL  # 初始化训练间隔

# 计算总资产值
def calculate_total_assets(portfolio, cash, current_prices):
    total_value = cash
    for stock, shares in portfolio.items():
        total_value += shares * current_prices[stock]
    return total_value

# 动态调整阈值函数
def update_thresholds(predicted_returns, base_buy_threshold, base_sell_threshold):
    mean_return = predicted_returns.mean()
    std_return = predicted_returns.std()

    # 打印调试信息
    print(f"Mean Predicted Return: {mean_return:.4f}, Std Dev: {std_return:.4f}")

    # 动态调整逻辑
    if std_return > 0:  # 确保标准差不为零
        buy_threshold = mean_return + std_return
        sell_threshold = mean_return - std_return / 2
    else:
        buy_threshold = base_buy_threshold
        sell_threshold = base_sell_threshold

    return max(buy_threshold, base_buy_threshold), min(sell_threshold, base_sell_threshold)

# 模型训练函数（包含保存模型逻辑）
def train_random_forest(features):
    print("Training Random Forest Model...")
    X = features[['mean_return', 'volatility']]
    y = features['total_return']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf_model = RandomForestRegressor(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    features['predicted_return'] = rf_model.predict(X)
    
    # 保存模型到文件
    model_filename = 'trained_rf_model.pkl'
    joblib.dump(rf_model, model_filename)
    print(f"Model saved to {model_filename}")
    
    return rf_model, features

# 模拟交易逻辑
print("Simulating trading...")
for time_step in tqdm(range(1, len(stock_data)), desc="Trading Simulation"):
    current_prices = stock_data.iloc[time_step]
    past_prices = stock_data.iloc[:time_step]  # 获取历史数据

    # 每 train_interval 时间步重新训练随机森林模型
    if time_step % train_interval == 0 or rf_model is None:
        # 准备特征数据
        returns = past_prices.pct_change().dropna()
        features = pd.DataFrame({
            "mean_return": returns.mean(),
            "volatility": returns.std(),
            "total_return": (past_prices.iloc[-1] - past_prices.iloc[0]) / past_prices.iloc[0]
        }).T
        features = features.T.reset_index()
        features.columns = ['Stock', 'mean_return', 'volatility', 'total_return']
        
        # 训练模型
        rf_model, features = train_random_forest(features)

        # 动态调整 BUY_THRESHOLD 和 SELL_THRESHOLD
        predicted_returns = features['predicted_return']
        BUY_THRESHOLD, SELL_THRESHOLD = update_thresholds(predicted_returns, BASE_BUY_THRESHOLD, BASE_SELL_THRESHOLD)
        print(f"Updated BUY_THRESHOLD: {BUY_THRESHOLD:.2f}, SELL_THRESHOLD: {SELL_THRESHOLD:.2f}")

        # 根据预测收益率排序，选择前20只股票
        features = features.sort_values(by='predicted_return', ascending=False)
        top_stocks = features.head(20)
        selected_stocks = top_stocks['Stock'].tolist()
        selected_stock_history.append(selected_stocks)

        # 当前可用现金
        available_cash = cash * 0.9  # 每次最多使用90%的现金

        # 遍历选中的股票
        for stock in selected_stocks:
            predicted_return = features.loc[features['Stock'] == stock, 'predicted_return'].values[0]
            
            # 买入逻辑
            if predicted_return > BUY_THRESHOLD and stock not in portfolio:
                buy_amount = available_cash * 0.05
                if buy_amount > cash:
                    buy_amount = cash
                shares_to_buy = buy_amount / current_prices[stock]
                portfolio[stock] = shares_to_buy
                cash -= buy_amount * (1 + TRANSACTION_COST)
                trade_log.append({
                    "time_step": time_step, "action": "buy", "stock": stock,
                    "amount": buy_amount, "price": current_prices[stock],
                    "remaining_cash": cash
                })
            
            # 卖出逻辑
            elif predicted_return < SELL_THRESHOLD and stock in portfolio:
                sell_value = portfolio[stock] * current_prices[stock]
                cash += sell_value * (1 - TRANSACTION_COST)
                del portfolio[stock]
                trade_log.append({
                    "time_step": time_step, "action": "sell", "stock": stock,
                    "amount": sell_value, "price": current_prices[stock],
                    "remaining_cash": cash
                })

    # 记录资产总值
    total_assets.append(calculate_total_assets(portfolio, cash, current_prices))

# 可视化部分
print("Generating visualizations...")
with tqdm(total=2, desc="Plotting Visualizations") as pbar:
    # 1. 总资产趋势与市场基准对比 - 采样 1/1000 的数据点
    sampled_indices = np.linspace(0, len(total_assets) - 1, num=len(total_assets) // 1000, dtype=int)
    sampled_total_assets = [total_assets[i] for i in sampled_indices]
    sampled_average_price = stock_data.iloc[sampled_indices].mean(axis=1)
    
    plt.figure(figsize=(14, 8))
    plt.plot(sampled_total_assets, label="Portfolio Value (Sampled)")
    plt.plot((sampled_average_price / sampled_average_price.iloc[0]) * INIT_FUND, 
             label="Market Benchmark (Sampled)", linestyle="--")
    plt.title("Portfolio vs Market Benchmark (Sampled Data)")
    plt.xlabel("Time Steps (Sampled)")
    plt.ylabel("Total Value")
    plt.legend()

    # 打印参数信息到图像上
    param_text = (
        f"INIT_FUND = {INIT_FUND:.2e}\n"
        f"BASE_BUY_THRESHOLD = {BASE_BUY_THRESHOLD}\n"
        f"BASE_SELL_THRESHOLD = {BASE_SELL_THRESHOLD}\n"
        f"TRANSACTION_COST = {TRANSACTION_COST}\n"
        f"BASE_TRAIN_INTERVAL = {BASE_TRAIN_INTERVAL}\n"
        f"N_ESTIMATORS = {N_ESTIMATORS}\n"
        f"MAX_DEPTH = {MAX_DEPTH}\n"
        f"WINDOW_SIZE = {WINDOW_SIZE}"
    )
    plt.text(0.02, 0.95, param_text, transform=plt.gca().transAxes, fontsize=10, 
             verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='lightgrey'))

    plt.show()
    pbar.update(1)

    # 3. 交易行为 - 保留完整交易日志数据
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

# 投资组合分布分析
if portfolio:
    # 计算最终投资组合分布
    final_portfolio_value = {stock: portfolio[stock] * stock_data.iloc[-1][stock] for stock in portfolio}
    total_value = sum(final_portfolio_value.values())
    portfolio_distribution = {stock: value / total_value for stock, value in final_portfolio_value.items()}
    
    # 绘制分布条形图
    plt.figure(figsize=(14, 8))
    plt.bar(portfolio_distribution.keys(), portfolio_distribution.values())
    plt.title("Final Portfolio Distribution")
    plt.xlabel("Stocks")
    plt.ylabel("Proportion of Total Value")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()
else:
    print("No stocks held in the final portfolio.")

# 回撤分析
def calculate_drawdown(total_assets):
    max_value = total_assets[0]
    drawdowns = []
    for value in total_assets:
        if value > max_value:
            max_value = value
        drawdown = (max_value - value) / max_value
        drawdowns.append(drawdown)
    return drawdowns, max(drawdowns)

# 计算回撤
drawdowns, max_drawdown = calculate_drawdown(total_assets)

# 绘制回撤图
plt.figure(figsize=(14, 8))
plt.plot(drawdowns, label="Drawdown")
plt.title(f"Drawdown Over Time (Max Drawdown: {max_drawdown:.2%})")
plt.xlabel("Time Steps")
plt.ylabel("Drawdown")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

print(f"Maximum Drawdown: {max_drawdown:.2%}")

# 敏感性分析
threshold_values = np.linspace(0.005, 0.03, 5)  # 调整阈值范围
results = []

for buy_threshold in threshold_values:
    # 使用不同的买入阈值模拟收益
    cash = INIT_FUND
    portfolio = {}
    total_assets_sensitivity = []

    for time_step in range(1, len(stock_data)):
        current_prices = stock_data.iloc[time_step]
        if time_step % train_interval == 0 or rf_model is None:
            returns = stock_data.iloc[:time_step].pct_change().dropna()
            features = pd.DataFrame({
                "mean_return": returns.mean(),
                "volatility": returns.std()
            }).reset_index(drop=True)
            rf_model = RandomForestRegressor(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, random_state=42, n_jobs=-1)
            rf_model.fit(features[['mean_return', 'volatility']], returns.mean(axis=1))
        predicted_returns = rf_model.predict(features[['mean_return', 'volatility']])
        
        for stock, predicted_return in zip(stock_data.columns, predicted_returns):
            if predicted_return > buy_threshold and stock not in portfolio:
                buy_amount = cash * 0.05
                shares_to_buy = buy_amount / current_prices[stock]
                portfolio[stock] = shares_to_buy
                cash -= buy_amount * (1 + TRANSACTION_COST)
            elif predicted_return < BASE_SELL_THRESHOLD and stock in portfolio:
                sell_value = portfolio[stock] * current_prices[stock]
                cash += sell_value * (1 - TRANSACTION_COST)
                del portfolio[stock]
        
        total_assets_sensitivity.append(calculate_total_assets(portfolio, cash, current_prices))
    
    final_assets_sensitivity = total_assets_sensitivity[-1]
    results.append((buy_threshold, final_assets_sensitivity))

# 绘制敏感性分析结果
thresholds, final_values = zip(*results)
plt.figure(figsize=(14, 8))
plt.plot(thresholds, final_values, marker='o')
plt.title("Sensitivity Analysis: Buy Threshold vs Final Portfolio Value")
plt.xlabel("Buy Threshold")
plt.ylabel("Final Portfolio Value")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()