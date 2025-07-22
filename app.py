import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
import plotly.graph_objects as go


# --- Cấu hình trang ---
st.set_page_config(
    page_title="ExRate - Model",
    page_icon="📈",
    layout="wide"
)


# --- Sidebar ---
st.sidebar.image("images/exchange-rate.png", width=150)
st.sidebar.markdown("### 1. Data")

# --- Tải dữ liệu mẫu ---
@st.cache_data
def load_sample_data():
    # Tạo dữ liệu giả định cho ví dụ
    # Trong thực tế, bạn sẽ tải dữ liệu từ file CSV, database, v.v.
    dates = pd.date_range(start='2017-01-01', periods=365*3, freq='D')
    sales = (
        100
        + 5 * np.sin(np.linspace(0, 30, len(dates))) * 10
        + np.random.normal(0, 5, len(dates))
        + np.linspace(0, 50, len(dates)) # xu hướng tăng
        + 20 * np.sin(np.linspace(0, 100, len(dates))) # seasonality
    ).astype(int)
    
    df = pd.DataFrame({'ds': dates, 'y': sales})
    return df

df = load_sample_data()

with st.sidebar.expander("Dataset"):
    st.write("Tải lên tập dữ liệu của bạn hoặc sử dụng dữ liệu mẫu.")
    uploaded_file = st.file_uploader("Chọn file CSV", type="csv")
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("Đã tải dữ liệu thành công!")
        except Exception as e:
            st.error(f"Lỗi khi tải file: {e}")

# Các mục khác trong sidebar có thể thêm vào:
st.sidebar.markdown("### 2. Modelling")
# st.sidebar.slider("Prior scale", min_value=0.01, max_value=10.0, value=0.5)
# st.sidebar.selectbox("Seasonalities", ["auto", "daily", "weekly", "yearly"])

# --- Main content ---
st.title("ExRate Model")

st.markdown("### What is this app?")
st.checkbox("Launch forecast", value=True)
st.checkbox("Track experiments")

st.markdown("## 1. Overview")
st.markdown("More info on this plot")

# Tạo mô hình Prophet và dự báo
m = Prophet(seasonality_mode='multiplicative')
m.fit(df)

future = m.make_future_dataframe(periods=365) # Dự báo 1 năm tiếp theo
forecast = m.predict(future)

# Biểu đồ dự báo
fig = plot_plotly(m, forecast)

# Thêm chú thích cho điểm dữ liệu cụ thể (như trong ảnh mẫu)
# Ở đây ta sẽ thêm một điểm chú thích giả định để minh họa
# Bạn có thể điều chỉnh để lấy dữ liệu thực tế từ forecast hoặc df
if not df.empty:
    sample_date = pd.to_datetime('2018-11-21') # Giả định một ngày để chú thích
    # Tìm giá trị gần nhất trong dữ liệu thực tế
    actual_data_on_sample_date = df[df['ds'] == sample_date]
    
    if not actual_data_on_sample_date.empty:
        # Lấy giá trị y (sales) từ dữ liệu thực tế
        actual_y = actual_data_on_sample_date['y'].iloc[0]
        
        # Thêm một annotation cho điểm này
        fig.add_trace(go.Scatter(
            x=[sample_date],
            y=[actual_y],
            mode='markers+text',
            text=[f"Nov 21, 2018, {actual_y} sales"],
            textposition="top center",
            marker=dict(size=10, color='red'),
            name="Sample Point"
        ))

st.plotly_chart(fig, use_container_width=True)

st.markdown("## 2. Forecast Components")
st.markdown("More info on these plots")
# Biểu đồ thành phần dự báo (trend, weekly, yearly)
fig_components = plot_components_plotly(m, forecast)
st.plotly_chart(fig_components, use_container_width=True)

# Hiển thị bảng dữ liệu (tùy chọn)
st.markdown("## 3. Raw Data")
st.dataframe(df.tail())
st.markdown("## 4. Forecast Data")
st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())