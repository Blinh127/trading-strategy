#!/usr/bin/env python
# coding: utf-8

# In[49]:


import vnstock
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from bayes_opt import BayesianOptimization
import datetime as dt
from vnstock import Vnstock
import talib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Danh sách cổ phiếu cần lấy dữ liệu
stocks =  ['HPG', 'VIB', 'MSN', 'MSH', 'EVF', 'CTG', 'MBB', 'KBC', 'SSI']

def get_stock_data(ticker):
    vnstock_instance = Vnstock().stock(symbol=ticker, source='VCI')
    start_date = "2000-01-01"
    end_date = "2025-01-01"
    df = vnstock_instance.quote.history(symbol=ticker, start=start_date, end=end_date)
    df['ticker'] = ticker
    return df

dataframes = [get_stock_data(stock) for stock in stocks]
df = pd.concat(dataframes, ignore_index=True)

def compute_indicators(df):
    df = df[['time', 'close', 'high', 'low', 'volume','ticker']].copy()
    df = df.sort_values(by=['ticker', 'time'])
    df['open'] = df.groupby('ticker')['close'].shift(1)
    df['rsi'] = talib.RSI(df['close'], timeperiod=14)
    df['macd'], df['macd_signal'], _ = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
    df['sma_10'] = talib.SMA(df['close'], timeperiod=10)
    df['sma_50'] = talib.SMA(df['close'], timeperiod=50)
    df['sma_200'] = talib.SMA(df['close'], timeperiod=200)
    df['sma_20'] = talib.SMA(df['close'], timeperiod=20)
    df['open'] = df['close'].shift(1)
    df['s10_s50_dist'] = (df['sma_10'] - df['sma_50']) / df['sma_50']
    df['s10_s50_dist_v'] = df['s10_s50_dist'] * df['volume']
    df['s50_s200_dist'] = (df['sma_50'] - df['sma_200']) / df['sma_200']
    df['s50_s200_dist_v'] = df['s50_s200_dist'] * df['volume']

    df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
    df['stoch_k'], df['stoch_d'] = talib.STOCH(df['high'], df['low'], df['close'], fastk_period=14, slowk_period=3, slowd_period=3)
    df['upper_band'], df['middle_band'], df['lower_band'] = talib.BBANDS(df['close'], timeperiod=20)
    df['bollinger_width'] = ((df['upper_band'] - df['lower_band']) / df['sma_20']) > 0.05
    df['adx'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
    df['cci'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=14)
    df['macd_signal_cut'] =(df['macd'] - df['macd_signal']) / df['macd_signal']
    df['macd_0230'] = ((df['macd'] - df['stoch_k'] )) /df['stoch_k']
    df['macd_0230_bb'] = df['macd_0230'] * df['bollinger_width']
    df['bullish_engulfing'] = (talib.CDLENGULFING(df['open'], df['high'], df['low'], df['close']) == 80).astype(int)
    df['bearish_engulfing'] = (talib.CDLENGULFING(df['open'], df['high'], df['low'], df['close']) == -80).astype(int)
    df['doji'] = talib.CDLDOJI(df['open'], df['high'], df['low'], df['close'])
    df['hammer'] = talib.CDLHAMMER(df['open'], df['high'], df['low'], df['close'])
    df['shooting_star'] = talib.CDLSHOOTINGSTAR(df['open'], df['high'], df['low'], df['close'])

    return df

df = df.groupby('ticker', group_keys=False).apply(compute_indicators)
df = df.reset_index(drop=True)

# 🛑 3. Loại bỏ dữ liệu không cần thiết + Sắp xếp theo thời gian
df = df[(df['time'] >= '2020-01-01') & (df['time'] <= '2025-01-01')]
df = df.sort_values(by="time") 

# 🛑 4. Tạo biến target DỰ ĐOÁN 5 ngày sau
df['future_max'] = df['close'].rolling(10).quantile(0.8).shift(-5)
df['target'] = ((df['future_max'] / df['close']) > 1.07).astype(int)

# 🛑 5. Chia train/val (KHÔNG TÍNH TOÁN LẠI FEATURE)
# Xác định ngưỡng thời gian để cắt train và val
split_date = "2024-01-01"  # VD: Train từ 2020-2022, Val từ 2023 trở đi

# Tạo train và val dựa trên thời gian
train_data = df[df['time'] < split_date].copy()
val_data = df[df['time'] >= split_date].copy()


# 🛑 6. Lấy feature và target
features = ['rsi', 'macd', 'macd_signal', 'sma_50', 'sma_10', 'sma_200', 'atr', 
            'stoch_k', 'stoch_d', 'middle_band', 'adx', 'cci', 
            'upper_band', 'lower_band', 'volume', 'high', 'low', 
            's10_s50_dist', 's50_s200_dist', 's10_s50_dist_v', 's50_s200_dist_v','macd_signal_cut', 'macd_0230_bb',
            'close','open', 'bullish_engulfing','bearish_engulfing','doji','hammer','shooting_star']



X_train, y_train = train_data[features], train_data['target']
X_val, y_val = val_data[features], val_data['target']

def check_data_leakage(X_train, X_val):
    # Kiểm tra trùng lặp
    duplicates = X_train.index.intersection(X_val.index)
    if len(duplicates) > 0:
        print(f"⚠️ Cảnh báo: Có {len(duplicates)} điểm dữ liệu bị trùng giữa tập train và validation!")
    else:
        print("✅ Không có dữ liệu trùng lặp giữa train và validation.")
    
    # Kiểm tra xem có dữ liệu tương lai xuất hiện trong train không
    train_dates = df.loc[X_train.index, 'time']
    val_dates = df.loc[X_val.index, 'time']
    if train_dates.max() >= val_dates.min():
        print("⚠️ Cảnh báo: Dữ liệu train có chứa thời gian vượt qua tập validation!")
    else:
        print("✅ Không có dữ liệu tương lai bị lẫn vào train.")
        
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  
X_val_scaled = scaler.transform(X_val)  


def xgb_evaluate(max_depth, min_child_weight,n_estimators, gamma, learning_rate, subsample, colsample_bytree, reg_lambda, alpha, scale_pos_weight):
    params = {
        'objective': 'binary:logistic',
        'max_depth': int(max_depth),
        'min_child_weight': min_child_weight,
        'n_estimators': int(n_estimators), 
        'gamma': gamma,
        'learning_rate': learning_rate,
        'subsample': subsample,
        'colsample_bytree': colsample_bytree,
        'reg_lambda': reg_lambda,
        'alpha': alpha,
        'scale_pos_weight': scale_pos_weight,
        'eval_metric': 'logloss'
    }
    model = xgb.XGBClassifier(**params, early_stopping_rounds=10)
    model.fit(X_train_scaled, y_train, eval_set=[(X_val_scaled, y_val)], verbose=False)
    y_pred_proba = model.predict_proba(X_val_scaled)[:, 1]
    logloss_value = log_loss(y_val, y_pred_proba)
    return -logloss_value

optimizer = BayesianOptimization(
    f=xgb_evaluate,
    pbounds={
        'max_depth': (30, 100),
        'min_child_weight': (4, 6),
        'n_estimators': (10, 1000),
        'gamma': (0, 0.05),
        'learning_rate': (0.05, 0.2),
        'subsample': (0.7, 1.0),
        'colsample_bytree': (0.5, 1.0),
        'reg_lambda': (5, 29),
        'alpha': (6, 30),
        'scale_pos_weight': (2.5, 3)
    },
    random_state=42
)
optimizer.maximize(init_points=5, n_iter=30)

best_params = optimizer.max['params']
best_params['max_depth'] = int(best_params['max_depth'])
best_params['n_estimators'] = int(best_params['n_estimators'])
print(best_params)

final_model = xgb.XGBClassifier(**best_params, objective='binary:logistic',early_stopping_rounds=10,  eval_metric='logloss')
final_model.fit(X_train_scaled, y_train, eval_set=[(X_train_scaled, y_train), (X_val_scaled, y_val)], verbose=True)

y_pred_proba = final_model.predict_proba(X_val_scaled)[:, 1]
y_train_proba = final_model.predict_proba(X_train_scaled)[:, 1]

# Dự đoán nhãn (0 hoặc 1) bằng cách chọn threshold = 0.5
y_pred = (y_pred_proba >= 0.94).astype(int)


# Tạo confusion matrix

# Tạo confusion matrix và chỉ định labels
cm = confusion_matrix(y_val, y_pred, labels=[0, 1])
print(cm)

# Tạo báo cáo đánh giá và chỉ định labels
report = classification_report(y_val, y_pred, labels=[0, 1], digits=4)
print(report)


logloss_value_val = log_loss(y_val, y_pred_proba)
logloss_value_train = log_loss(y_train, y_train_proba)

print(f"Validation Logloss: {logloss_value_val:.4f}")
print(f"Training Logloss: {logloss_value_train:.4f}")
    

evals_result = final_model.evals_result()
plt.figure(figsize=(10, 5))
plt.plot(evals_result['validation_0']['logloss'], label='Train Logloss', color='blue')
plt.plot(evals_result['validation_1']['logloss'], label='Validation Logloss', color='red')
plt.xlabel('Iterations')
plt.ylabel('Logloss')
plt.title('Train vs Validation Logloss')
plt.legend()
plt.show()


# In[50]:


import matplotlib.pyplot as plt

y_rolling = df['close'].rolling(10).max().shift(-5)
y_shift = df['close'].shift(-5)
plt.hist(y_rolling, bins=50, alpha=0.5, label="rolling(10).max().shift(-5)")
plt.hist(y_shift, bins=50, alpha=0.5, label="shift(-5)")
plt.legend()
plt.show()


# In[65]:


import matplotlib.pyplot as plt

# Vẽ biểu đồ giá với điểm mua cách nhau ít nhất 10 ngày
for stock in stocks:
    stock_df = val_data[val_data['ticker'] == stock].copy()
    stock_df['predicted_buy'] = y_pred[val_data['ticker'] == stock]

    # Đảm bảo các tín hiệu mua cách nhau ít nhất 10 ngày
    buy_signals = stock_df[stock_df['predicted_buy'] == 1].index
    filtered_buy_signals = []
    
    last_buy_index = -10  # Đảm bảo tín hiệu đầu tiên được xét
    for idx in buy_signals:
        if idx >= last_buy_index + 10:
            filtered_buy_signals.append(idx)
            last_buy_index = idx
    
    # Đánh dấu lại các tín hiệu mua hợp lệ
    stock_df['filtered_buy'] = 0
    stock_df.loc[filtered_buy_signals, 'filtered_buy'] = 1

    # Tính toán các điểm False Positive cho cổ phiếu này
    false_positive_indices = (y_pred[val_data['ticker'] == stock] == 1) & (y_val[val_data['ticker'] == stock] == 0)
    false_positive_points = stock_df[false_positive_indices]

    # các chỉ báo đảo chiều 
    

    # Vẽ biểu đồ
    plt.figure(figsize=(12, 6))
    plt.plot(stock_df['time'], stock_df['close'], label='Close Price', color='blue')
    plt.scatter(stock_df['time'][stock_df['filtered_buy'] == 1], 
                stock_df['close'][stock_df['filtered_buy'] == 1], 
                color='yellow', label='Buy Signal', marker='o')
    plt.scatter(false_positive_points['time'], false_positive_points['close'], 
                color='red', label='False Positive', marker='x', s=100)
    
    plt.title(f'Stock Price and Buy Signals - {stock}')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.xticks(rotation=45)
    plt.show()


# In[52]:


plt.figure(figsize=(10, 5))

# Vẽ histogram phân phối xác suất dự đoán trên tập validation
plt.hist(y_pred_proba, bins=50, alpha=0.7, color='blue', edgecolor='black', label="Validation Predictions")

# Vẽ histogram phân phối xác suất trên tập train để so sánh
plt.hist(y_train_proba, bins=50, alpha=0.7, color='red', edgecolor='black', label="Training Predictions")

plt.axvline(0.5, color='black', linestyle='dashed', label="Threshold = 0.5")
plt.axvline(0.59, color='green', linestyle='dashed', label="Threshold = 0.59")

plt.xlabel("Predicted Probability")
plt.ylabel("Frequency")
plt.title("Distribution of Predicted Probabilities")
plt.legend()
plt.grid()
plt.show()


# In[53]:


import xgboost as xgb
import matplotlib.pyplot as plt


xgb.plot_importance(final_model, importance_type="gain")  # "gain" quan trọng hơn "weight"
plt.show()


# In[54]:


import shap

# Tính giá trị SHAP
explainer = shap.Explainer(final_model, X_train_scaled)
shap_values = explainer(X_val_scaled)

# Vẽ biểu đồ SHAP summary plot
shap.summary_plot(shap_values, X_val)


# In[55]:


import numpy as np

# Xác định các điểm cực đoan
extreme_train = np.sum((y_train_proba < 0.05) | (y_train_proba > 0.95))
extreme_valid = np.sum((y_pred_proba < 0.05) | (y_pred_proba > 0.95))

# Tính tỷ lệ phần trăm
total_train = len(y_train_proba)
total_valid = len(y_pred_proba)

percent_train = (extreme_train / total_train) * 100
percent_valid = (extreme_valid / total_valid) * 100

print(f"Số lượng điểm cực đoan (Train): {extreme_train} / {total_train} ({percent_train:.2f}%)")
print(f"Số lượng điểm cực đoan (Validation): {extreme_valid} / {total_valid} ({percent_valid:.2f}%)")


# In[56]:


extreme_train_idx = np.where((y_train_proba < 0.05) | (y_train_proba > 0.95))[0]
extreme_valid_idx = np.where((y_pred_proba < 0.05) | (y_pred_proba > 0.95))[0]

print("Extreme Train Samples:")
print(X_train.iloc[extreme_train_idx])  # Hiển thị các đặc trưng của điểm cực đoan
print("\nExtreme Validation Samples:")
print(X_val.iloc[extreme_valid_idx])

print ( len(extreme_train_idx))
print ( len(extreme_valid_idx))


# In[57]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np


def optimize_threshold(y_true, y_prob):
    best_threshold = 0.5  # Default value
    best_metrics = (0, 0, 0, 0)  # (accuracy, precision, recall, f1)
    
    thresholds = np.linspace(0, 1, 100)  # Duyệt qua 100 ngưỡng từ 0 đến 1
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)  # Chuyển xác suất thành nhãn dự đoán
        
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        
        # Kiểm tra điều kiện: tất cả metric phải >= 70%
        if  acc > 0.7 and f1 > 0.7 and prec > 0.7 and rec > 0.7:
            best_threshold = threshold
            best_metrics = (acc, prec, rec, f1)
            
    return best_threshold, best_metrics

# Áp dụng cho dữ liệu thực tế
y_true = y_val  # Nhãn thực tế từ tập validation
y_prob = y_pred_proba  # Xác suất dự đoán từ mô hình

best_threshold, best_metrics = optimize_threshold(y_true, y_prob)
print(f"Best Threshold: {best_threshold}, Metrics: {best_metrics}")

# Dự đoán lại với ngưỡng tối ưu
y_pred_optimized = (y_prob >= best_threshold).astype(int)
# Tạo confusion matrix và chỉ định labels

cm_optimized = confusion_matrix(y_true, y_pred_optimized,labels=[0, 1] )
print("Optimized Confusion Matrix:")
print(cm_optimized)

# In báo cáo đánh giá tối ưu
report_optimized = classification_report(y_true, y_pred_optimized,labels=[0, 1], digits=4)
print("Optimized Classification Report:")
print(report_optimized)


# In[62]:


import matplotlib.pyplot as plt
import pandas as pd
import talib

# Giả sử bạn đã có DataFrame 'val_data' với các cột: 'time', 'ticker', 'open', 'close'
# 'y_pred' là kết quả dự đoán của mô hình cho điểm mua

# Hàm để tính toán các tín hiệu đảo chiều
def detect_reversal_patterns(df):
    # Các mô hình nến đảo chiều cơ bản

    df['engulfing'] = talib.CDLENGULFING(df['open'], df['high'], df['low'], df['close'])
    
    # Bullish Engulfing (+100)
    df['bullish_engulfing'] = df['engulfing'].apply(lambda x: 1 if x > 0 else 0)
    
    # Bearish Engulfing (-100)
    df['bearish_engulfing'] = df['engulfing'].apply(lambda x: -1 if x < 0 else 0)
   
    return df

# Dữ liệu của bạn
for stock in stocks:
    stock_df = val_data[val_data['ticker'] == stock].copy()
    stock_df['predicted_buy'] = y_pred[val_data['ticker'] == stock]

    # Đảm bảo các tín hiệu mua cách nhau ít nhất 10 ngày
    buy_signals = stock_df[stock_df['predicted_buy'] == 1].index
    filtered_buy_signals = []
    
    last_buy_index = -10  # Đảm bảo tín hiệu đầu tiên được xét
    for idx in buy_signals:
        if idx >= last_buy_index + 10:
            filtered_buy_signals.append(idx)
            last_buy_index = idx
    
    # Đánh dấu lại các tín hiệu mua hợp lệ
    stock_df['filtered_buy'] = 0
    stock_df.loc[filtered_buy_signals, 'filtered_buy'] = 1

    # Tính toán các điểm False Positive cho cổ phiếu này
    
    # Các chỉ báo đảo chiều
    stock_df = detect_reversal_patterns(stock_df)

    # Lọc các tín hiệu đảo chiều
    bullish_signals = stock_df[stock_df['bullish_engulfing'] == 1]
    bearish_signals = stock_df[stock_df['bearish_engulfing'] == -1]
     # Tìm các tín hiệu bullish và bearish trùng với các tín hiệu mua hợp lệ

    # In ra các tín hiệu đảo chiều
    print(f"{stock} Bullish Engulfing Signals:")
    print(bullish_signals[['time', 'bullish_engulfing']])
    
    print(f"{stock} Bearish Engulfing Signals:")
    print(bearish_signals[['time', 'bearish_engulfing']])
    
    # Vẽ biểu đồ cho cổ phiếu này
    plt.figure(figsize=(12, 6))
    plt.plot(stock_df['time'], stock_df['close'], label='Close Price', color='blue')
    
    # Vẽ các tín hiệu Bullish Engulfing
    #plt.scatter(bullish_signals['time'], bullish_signals['close'], color='green', label='Bullish Engulfing', marker='o')
    
    # Vẽ các tín hiệu Bearish Engulfing
    #plt.scatter(bearish_signals['time'], bearish_signals['close'], color='red', label='Bearish Engulfing', marker='x')
    
    # Vẽ các tín hiệu mua hợp lệ (filtered_buy)
    plt.scatter(stock_df['time'][stock_df['filtered_buy'] == 1], 
                stock_df['close'][stock_df['filtered_buy'] == 1], 
                color='yellow', label='Buy Signal', marker='o')
    
    # Vẽ các điểm False Positive

    plt.scatter(bullish_signals['time'], bullish_signals['close'] , color='lime', label='Bullish Engulfing (Mua?)', marker='o')
    plt.scatter(bearish_signals['time'], bearish_signals['close'] , color='black', label='Bearish Engulfing (Bán?)', marker='x')
    #plt.scatter(bullish_signals['time'], bullish_signals['close'], 
    #        color='green', label='Bullish Engulfing (Mua?)', marker='o', zorder=3)

    # Bearish Engulfing (Màu đen)
    #plt.scatter(bearish_signals['time'], bearish_signals['close'], 
    #        color='black', label='Bearish Engulfing (Bán?)', marker='x', zorder=3)

   
    plt.title(f'Stock Price and Buy Signals - {stock}')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.xticks(rotation=45)
    plt.show()


# In[59]:


print(df[['bullish_engulfing', 'bearish_engulfing', 'doji', 'hammer', 'shooting_star']].sum())



# In[63]:


print(stock_df[['time', 'open', 'high', 'low', 'close', 'bullish_engulfing']].head(10))


# In[ ]:




