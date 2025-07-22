import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
def loadData():
    # --- 1. Thu thập dữ liệu ---
    # Sử dụng yfinance để tải dữ liệu USD/VND (sử dụng mã cặp tiền VND=X)
    # Cần điều chỉnh khoảng thời gian phù hợp
    ticker_symbol = "VND=X" # Mã cặp tiền USD/VND trên Yahoo Finance
    start_date = "2019-01-01"
    # lấy dữ liệu đến ngày hôm qua
    end_date = datetime.now() - timedelta(days=1)
    end_date = end_date.strftime("%Y-%m-%d")  # Chuyển đổi định dạng ngày tháng

    try:
        data = yf.download(ticker_symbol, start=start_date, end=end_date)
        if data.empty:
            print(f"Không có dữ liệu cho mã '{ticker_symbol}' trong khoảng thời gian đã chọn. Vui lòng kiểm tra lại mã hoặc khoảng thời gian.")
            exit()
    except Exception as e:
        print(f"Lỗi khi tải dữ liệu: {e}. Vui lòng kiểm tra kết nối internet hoặc mã cặp tiền.")
        exit()
    data = data[['Close']].copy()
    data.rename(columns={'Close': 'USDVND'}, inplace=True)
    # Tạo 1 dataframe có 2 cột là ds và y
    df = pd.DataFrame({
        'date': data.index,
        'USDVND': data[['USDVND']].values.flatten()
    })
    print(df)
    return df
