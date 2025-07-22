import numpy as np
from arch import arch_model
import pandas as pd

# --- 2. Tính toán lợi suất Logarithmic ---
def calculate_log_returns(df):
    df['log_return'] = np.log(df['USDVND'] / df['USDVND'].shift(1))
    df.dropna(inplace=True)  # Loại bỏ giá trị NaN
    return df

def find_best_garch_model(df,model_type, distribution, p_max=3, q_max=3):
    best_aic = np.inf
    best_order = None
    best_model = None
    
    if model_type == "ARCH":
        vol = 'Arch'
    elif model_type == "GARCH":
        vol = 'Garch'
    elif model_type == "EGARCH":
        vol = 'EGarch'
    elif model_type == "TARCH":
        vol = 'Tarch'
    if distribution == "Normal":
        dist = 'normal'
    elif distribution == "Student's t":
        dist = 't'
    elif distribution == "Skewed Student's t":
        dist = 'skewt'
    # Tìm mô hình GARCH với các giá trị p, q từ 0 đến p_max, q_max
    for p in range(4):
        for q in range(4):
            try:
                model = arch_model(df, vol=vol, p=p, q=q, dist=dist)
                model_fit = model.fit(disp='off')
                if model_fit.aic < best_aic:
                    best_aic = model_fit.aic
                    best_order = (p, q)
                    best_model = model_fit
            except Exception as e:
                continue

    return best_model, best_order

