import numpy as np
# --- 2. Tính toán lợi suất Logarithmic ---
def calculate_log_returns(df):
    df['log_return'] = np.log(df['USDVND'] / df['USDVND'].shift(1))
    df.dropna(inplace=True)  # Loại bỏ giá trị NaN
    return df