import pandas as pd
import numpy as np
from vnstock import Vnstock
import matplotlib.pyplot as plt
import datetime as dt
from matplotlib.patches import Ellipse
from sklearn.linear_model import LinearRegression

stocks = ['HPG', 'VIB', 'CTG','EVF','VGC','HHV','PHP','MBB','VN30F1M']

# Ngày bắt đầu và kết thúc lấy dữ liệu
end_date = dt.datetime.now().strftime('%Y-%m-%d')
#dt.datetime.now().strftime('%Y-%m-%d')
start_date =  (dt.datetime.now() - dt.timedelta(days=1000)).strftime('%Y-%m-%d')

vnstock_instance = Vnstock()

# Lấy dữ liệu lịch sử
historical_data = {}
for stock in stocks:
    try:
        df = vnstock_instance.stock(symbol=stock, source='VCI').quote.history(start=start_date, end=end_date)
        df['date'] = pd.to_datetime(df['time'])
        df.set_index('date', inplace=True)
        df.sort_index(inplace=True)
        historical_data[stock] = df.copy()
    except Exception as e:
        print(f"Lỗi khi lấy dữ liệu {stock}: {e}")
        
def trendline_regression(df, column, window):
    # Ensure 'column' exists in df
    if column not in df.columns:
        print(f"Column '{column}' not found in DataFrame")
        return None, None, None

    recent_data = df[column].dropna().tail(window)
    if len(recent_data) < 2:
        return None, None, None  # Not enough data for regression

    x = np.arange(len(recent_data)).reshape(-1, 1)
    y = recent_data.values.reshape(-1, 1)

    model = LinearRegression().fit(x, y)
    trendline = model.predict(x)
    r_squared = model.score(x, y)

    return recent_data.index, trendline.flatten(), r_squared

# Tìm bộ `window` tối ưu dựa trên R² của giá Close
def find_best_window(stock, days=1000, max_window=1000, min_window=100):
    best_r2 = -1.0  # R² cao nhất
    best_window = None
    best_df = None
    
    try:
        df = vnstock_instance.stock(symbol=stock, source='VCI').quote.history(start=start_date, end=end_date)
        df['date'] = pd.to_datetime(df['time'])
        df.set_index('date', inplace=True)
        df.sort_index(inplace=True)

        for window in range(min_window, max_window + 1, 5):  # Duyệt window (100, 105, ..., 1000)
            _, _, r2 = trendline_regression(df, 'close', window)
            
            if r2 is not None and r2 > best_r2:  # Chọn bộ có R² cao nhất
                best_r2 = r2
                best_window = window
                best_df = df

    except Exception as e:
        print(f"Lỗi khi lấy dữ liệu {stock}: {e}")

    return best_window, best_r2, best_df  # Trả về bộ tối ưu nhất


print(df.head())
# Hàm tính MACD
def calculate_macd(df, short_window=12, long_window=26, signal_window=9):
    df['EMA12'] = df['close'].ewm(span=short_window, adjust=False).mean()
    df['EMA26'] = df['close'].ewm(span=long_window, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['Signal_Line'] = df['MACD'].ewm(span=signal_window, adjust=False).mean()
    return df

# Hàm tính Stochastic RSI
def calculate_stoch_rsi(df, period=14, smoothK=3, smoothD=3):
    delta = df['close'].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=period).mean()
    avg_loss = pd.Series(loss).rolling(window=period).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['RSI_Low'] = df['RSI'].rolling(window=period).min()
    df['RSI_High'] = df['RSI'].rolling(window=period).max()
    df['Stoch_RSI'] = 100 * ((df['RSI'] - df['RSI_Low']) / (df['RSI_High'] - df['RSI_Low']))
    df['Stoch_RSI_K'] = df['Stoch_RSI'].rolling(window=smoothK).mean()
    df['Stoch_RSI_D'] = df['Stoch_RSI_K'].rolling(window=smoothD).mean()
    return df

# Hàm xác định điểm Mua/Bán tối ưu
def find_trade_signals(df):
    df['Buy_Signal'] = False
    df['Sell_Signal'] = False
    df['Invalid_Buy_Zone'] = False
    df['Invalid_Sell_Zone'] = False
    last_valid_buy = None
    last_valid_sell = None

    for i in range(1, len(df)):
        downtrend = df['MACD'][i] <= df['Signal_Line'][i]
        uptrend = df['MACD'][i] >= df['Signal_Line'][i]

        if df['MACD'][i] <= -0.2 and df['Stoch_RSI_K'][i] <= 20 and df['Stoch_RSI_D'][i] <=20 and downtrend:
            if df['MACD'][i] <= df['MACD'][i - 1] and df['Signal_Line'][i] <= df['Signal_Line'][i - 1]:
                last_valid_buy = i
            if last_valid_buy is not None:
                df.at[last_valid_buy, 'Buy_Signal'] = True
            else:
                df.at[i, 'Invalid_Buy_Zone'] = True

        if df['MACD'][i] >= 0.2 and df['Stoch_RSI_K'][i] >= 80 and df['Stoch_RSI_D'][i] >= 80 and uptrend:
            if df['MACD'][i] >= df['MACD'][i - 1] and df['Signal_Line'][i] >= df['Signal_Line'][i - 1]:
                last_valid_sell = i
            elif last_valid_sell is not None:
                df.at[last_valid_sell, 'Sell_Signal'] = True

    return df

# Hàm lấy dữ liệu và thực hiện phân tích
def analyze_stock(symbol):
    vnstock_instance = Vnstock().stock(symbol='ACB', source='VCI') 
    end_date = dt.datetime.now().strftime('%Y-%m-%d')
    start_date =  (dt.datetime.now() - dt.timedelta(days=1000)).strftime('%Y-%m-%d')
    df = vnstock_instance.quote.history(symbol=stock, start=start_date, end=end_date)
    df = df[['time', 'close', 'high', 'low', 'open', 'volume']].rename(columns={'time': 'Date'})
    df['Date'] = pd.to_datetime(df['Date'])
    
    
    df = calculate_macd(df)
    df = calculate_stoch_rsi(df)
    df = find_trade_signals(df)

    
    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax1.plot(df.index, df['close'], label=f'Giá {stock}', color='blue', linewidth=1)
    buy_points = df[df['Buy_Signal']]
    sell_points = df[df['Sell_Signal']]
    ax1.scatter(buy_points.index, buy_points['close'], color='green', label='Buy Signal', marker='^', s=100)
    ax1.scatter(sell_points.index, sell_points['close'], color='red', label='Sell Signal', marker='v', s=100)

    window_opt, best_r2, _ = find_best_window(stock)
    trend_dates_c, trend_c, r2_c = trendline_regression(df, 'close', window_opt)
    trend_dates_h, trend_h, r2_h = trendline_regression(df, 'high', window_opt)
    trend_dates_l, trend_l, r2_l = trendline_regression(df, 'low', window_opt)

    for i in range(len(df)):
        if df['Invalid_Buy_Zone'][i]:
            ax1.add_patch(Ellipse((df.index[i], df['close'].iloc[i]), 5, 5, color='cyan', alpha=0.3))
        if df['Invalid_Sell_Zone'][i]:
            ax1.add_patch(Ellipse((df.index[i], df['close'].iloc[i]), 5, 5, color='magenta', alpha=0.3))
    if len(trend_c) > 0:
            ax1.plot(trend_dates_c, trend_c, color="red", linestyle="dashed", label=f"close Trend (R²={r2_c:.2f})")
    if len(trend_h) > 0:
            ax1.plot(trend_dates_h, trend_h, color="orange", linestyle="dashed", label=f"High Trend (R²={r2_h:.2f})")
    if len(trend_l) > 0:
           ax1.plot(trend_dates_l, trend_l, color="green", linestyle="dashed", label=f"Low Trend (R²={r2_l:.2f})")
    ax1.set_xlabel('Ngày')
    ax1.set_ylabel('Giá đóng cửa')
    ax1.set_title(f'Tín hiệu giao dịch cho {stock}')
    ax1.legend()
    ax1.grid()
    
    #ax2 = ax1.twinx()
    #ax2.plot(df.index, df['MACD'], label='MACD', color='purple', linestyle='dashed')
    #ax2.plot(df.index, df['Signal_Line'], label='Signal Line', color='orange', linestyle='dashed')
    #ax2.set_ylabel('MACD')
    #ax2.legend(loc='upper left')
    
    plt.show()
    
    return df

stocks = ['HPG', 'EVF', 'VIB', 'MML', 'PLC', 'CTG','VN30F1M']
for stock in stocks:
    analyze_stock(stock)
