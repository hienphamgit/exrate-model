import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
import plotly.graph_objects as go
import loaddata as ld
import warnings
import buildmodel as bm
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
from arch import arch_model

warnings.filterwarnings("ignore")




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
def get_data():
    return ld.loadData()
# Hiển thị loading khi tải dữ liệu
with st.spinner("Đang tải dữ liệu..."):
    df = get_data()

with st.sidebar.expander("Dataset"):
    st.write("Tải lên tập dữ liệu hoặc sử dụng dữ liệu mẫu (Hiên tại chỉ hỗ trợ phân tích dữ liệu theo ngày). Chú ý: Dữ liệu phải có cột 'date' và 'USDVND'.")
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
st.title("Mô hình dự báo tỷ giá hối đoái USD/VND")

st.markdown("## 1. Trực quan hoá dữ liệu")
st.markdown("Dữ liệu tỷ giá hối đoái USD/VND từ 2019 đến nay được thu thập từ Yahoo Finance. ")
# Vẽ biểu đồ đường tỷ giá hối đoái USD/VND
fig = go.Figure()
fig.add_trace(go.Scatter(x=df['date'], y=df['USDVND'], mode='lines', name='USD/VND'))
fig.update_layout(title='Tỷ giá hối đoái USD/VND theo ngày', xaxis_title='', yaxis_title='Tỷ giá (VND)')
st.plotly_chart(fig, use_container_width=True)

# Hiển thị mô tả thống kê của dữ liệu, căn chỉnh bảng và tiêu đề cho đẹp
st.markdown("### Mô tả thống kê dữ liệu")
st.dataframe(df.describe().transpose(), use_container_width=True)

# --- 2. Tính toán lợi suất Logarithmic ---
st.markdown("## 2. Phân tích lợi suất Logarithmic")
df = bm.calculate_log_returns(df)
col1, col2 = st.columns(2)
# --- Biểu đồ 1: Lợi suất Logarithmic theo thời gian (Column 1) ---
with col1:
    st.subheader("Lợi suất Logarithmic của USD/VND theo thời gian")
    fig1 = go.Figure()
    fig1.add_trace(
        go.Scatter(x=df['date'], y=df['log_return'], mode='lines', name='Lợi suất Log', line=dict(color='blue'))
    )
    fig1.update_layout(
        xaxis_title="",
        yaxis_title="Lợi suất Logarithmic",
        height=400,
        margin=dict(l=40, r=20, t=40, b=40), # Điều chỉnh lề
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='LightGrey'),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='LightGrey')
    )
    st.plotly_chart(fig1, use_container_width=True)

# --- Biểu đồ 2: Phân phối lợi suất Logarithmic (Column 2) ---
with col2:
    st.subheader("Phân phối lợi suất Logarithmic")
    fig2 = go.Figure()

    # Histogram Trace
    hist_trace = go.Histogram(
        x=df['log_return'],
        nbinsx=50,
        name='Lợi suất Log',
        marker_color='lightblue',
        opacity=0.8,
        histnorm='probability density',
        showlegend=False
    )
    fig2.add_trace(hist_trace)

    # Tính toán và thêm KDE Trace
    x_kde = np.linspace(df['log_return'].min(), df['log_return'].max(), 500)
    kde = gaussian_kde(df['log_return'])
    kde_y = kde(x_kde)

    kde_trace = go.Scatter(
        x=x_kde,
        y=kde_y,
        mode='lines',
        name='KDE',
        line=dict(color='red', dash='dash', width=2),
        showlegend=False
    )
    fig2.add_trace(kde_trace)

    fig2.update_layout(
        xaxis_title="",
        yaxis_title="",
        height=400,
        margin=dict(l=40, r=20, t=40, b=40), # Điều chỉnh lề
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='LightGrey'),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='LightGrey')
    )
    st.plotly_chart(fig2, use_container_width=True)

# --------------- Xây dựng mô hình dự báo tỷ giá hối đoái ----------------------
st.markdown("## 3. Dự báo tỷ giá hối đoái USD/VND")

col3, col4 = st.columns(2)

with col3:
    # Widget chọn khoảng thời gian dự báo trong cột 1
    forecast_period = st.slider(
        "Chọn khoảng thời gian dự báo (ngày)",
        min_value=1,
        max_value=365,
        value=30
    )
    
    # commbox chọn phân phối cho mô hình GARCH
    distribution = st.selectbox(
        "Chọn phân phối cho mô hình GARCH",
        options=["Normal", "Student's t", "Skewed Student's t"],
        index=0  # Mặc định chọn Normal
    )
    # Chọn loại mô hình ARCH, GARCH, EGARCH, TARCH
    model_type = st.selectbox(
        "Chọn loại mô hình ARCH/GARCH",
        options=["ARCH", "GARCH", "EGARCH", "TARCH"],
        index=1  # Mặc định chọn GARCH
    )
with col4:
    # Thêm radio button để chọn tìm mô hình hiệu quả nhất hoặc manually chọn mô hình. Nếu chọn manually thì sẽ hiện ra ô nhập giá trị p, q
    model_selection = st.radio(
        "Chọn cách xây dựng mô hình",
        options=["Tự động tìm mô hình tốt nhất", "Tùy chỉnh mô hình"],
        index=1  # Mặc định chọn Tự động tìm mô hình tốt nhất
    )
    # Nếu chọn "Chọn mô hình theo p, q" thì hiển thị ô nhập loại mô hình và giá trị p, q
    if model_selection == "Tùy chỉnh mô hình":
        p = st.number_input("Nhập giá trị p (0-3)", min_value=0, max_value=3, value=1)
        q = st.number_input("Nhập giá trị q (0-3)", min_value=0, max_value=3, value=1)


if model_selection == "Tự động tìm mô hình tốt nhất":
    # Tự động tìm mô hình tốt nhất
    with st.spinner("Đang tìm mô hình tốt nhất..."):
        best_model, best_params = bm.find_best_garch_model(df['log_return'], model_type, distribution, forecast_period)
        if best_model is None:
            st.error("Không tìm thấy mô hình GARCH phù hợp.")
        else:
            st.success(f"Đã tìm thấy mô hình tốt nhất với tham số (p,q) = {best_params}")
   
else:
    # Tạo mô hình với các tham số tùy chỉnh
    with st.spinner("Đang xây dựng mô hình với tham số tùy chỉnh..."):
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
        try:
            best_model = arch_model(df['log_return'], vol=vol, p=p, q=q, dist=dist)
            best_model = best_model.fit(disp='off')
            st.success(f"Đã xây dựng mô hình với tham số (p,q) = ({p},{q})")
        except Exception as e:
            st.error(f"Lỗi khi xây dựng mô hình: {e}")
            best_model = None
if best_model is not None:
    #Hiển thị thông tin mô hình
    st.write(best_model.summary())
    col9, col0 = st.columns(2)
    with col9:
        st.markdown("### Thông tin mô hình GARCH")
        st.write(f"Mô hình: {best_model.model.__class__.__name__}")
        st.write(f"AIC: {best_model.aic:.2f}")
        st.write(f"BIC: {best_model.bic:.2f}")
        st.write(f"Log-Likelihood: {best_model.loglikelihood:.2f}")
    with col0:
        fig_garch = best_model.plot(annualize='D')
        st.pyplot(fig_garch)

    #Chia làm 2 cột để hiển thị kết quả dự báo
    col5, col6 = st.columns(2)
    with col5:
            # kiểm định phần dư của mô hình
        residuals = best_model.resid
        st.subheader("Kiểm định phần dư của mô hình")
        fig_residuals = go.Figure()
        fig_residuals.add_trace(go.Scatter(
            x=df['date'],
            y=residuals,
            mode='lines',
            name='Phần dư',
            line=dict(color='blue')
        ))
        fig_residuals.update_layout(
            title='',
            xaxis_title='',
            yaxis_title='Phần dư',
            height=400,
            margin=dict(l=40, r=20, t=40, b=40),  # Điều chỉnh lề
            xaxis=dict(showgrid=True, gridwidth=1, gridcolor='LightGrey'),
            yaxis=dict(showgrid=True, gridwidth=1, gridcolor='LightGrey')
        )
        st.plotly_chart(fig_residuals, use_container_width=True)
    with col6:
        # Hiển thị phân phối phần dư
        st.subheader("Phân phối phần dư của mô hình")
        fig_residuals_dist = go.Figure()
        fig_residuals_dist.add_trace(go.Histogram(
            x=residuals,
            nbinsx=50,
            name='Phần dư',
            marker_color='lightblue',
            opacity=0.8,
            histnorm='probability density',
            showlegend=False
        ))
        # Tính toán và thêm KDE Trace
        x_kde = np.linspace(residuals.min(), residuals.max(), 500)
        kde = gaussian_kde(residuals)
        kde_y = kde(x_kde)
        kde_trace = go.Scatter(
            x=x_kde,
            y=kde_y,
            mode='lines',
            name='KDE',
            line=dict(color='red', dash='dash', width=2),
            showlegend=False
        )
        fig_residuals_dist.add_trace(kde_trace)
        fig_residuals_dist.update_layout(
            xaxis_title='',
            yaxis_title='',
            height=400,
            margin=dict(l=40, r=20, t=40, b=40),  # Điều chỉnh lề
            xaxis=dict(showgrid=True, gridwidth=1, gridcolor='LightGrey'),
            yaxis=dict(showgrid=True, gridwidth=1, gridcolor='LightGrey')
        )
        st.plotly_chart(fig_residuals_dist, use_container_width=True)
    if model_type != "EGARCH":
        # Dự báo với mô hình tốt nhất
        forecast = best_model.forecast(horizon=forecast_period)
        conditional_variance_forecast = forecast.variance.iloc[-1]
        volatility_forecast = np.sqrt(conditional_variance_forecast)
        st.subheader("Dự báo biến động tỷ giá trong tương lai")
        # --- 2. Dự báo giá trị (Lợi suất và Tỷ giá) ---
        # Dự báo lợi suất (mean forecast)
        mean_forecast = forecast.mean.iloc[-1]
        # Tính giá trị tỷ giá dự báo
        # Giá tỷ giá cuối cùng từ dữ liệu gốc
        last_price = df['USDVND'].iloc[-1]
        last_date = df["date"].iloc[-1] 
            # 'B' là tần số ngày làm việc (business day)
        forecast_dates = pd.date_range(start=last_date, periods=forecast_period + 1, freq='B')[1:]

        # Tạo DataFrame để lưu trữ giá dự báo
        forecasted_prices = pd.DataFrame(index=forecast_dates) # Gán index ngày tháng ngay lập tức
        forecasted_prices['Forecasted_USDVND'] = np.nan
        # thêm cột biến động dự báo
        forecasted_prices['Forecasted_Volatility'] = np.nan
        current_forecast_price = last_price

        for i, log_return_pred in enumerate(mean_forecast.values):
            current_forecast_price = current_forecast_price * np.exp(log_return_pred)
            forecasted_prices.iloc[i, 1] = volatility_forecast[i]
            forecasted_prices.iloc[i, 0] = current_forecast_price
            

        # tạo 2 cột để hiển thị giá trị dự báo và vẽ biểu đồ
        col7, col8 = st.columns(2)

        with col7:
            st.write(f"Giá tỷ giá USD/VND dự báo trong {forecast_period} ngày tới:")
            st.dataframe(forecasted_prices)
        with col8:
            st.write("")
        # cắt bớt dữ liệu df 1 năm gần nhất để vẽ biểu đồ
        df = df[df['date'] >= (df['date'].max() - pd.DateOffset(years=1))]
        # Vẽ biểu đồ tỷ giá lịch sử và tỷ giá dự báo với tỷ giá dự báo nét đứt
        st.subheader("Biểu đồ tỷ giá USD/VND dự báo")
        fig_forecast = go.Figure()
        fig_forecast.add_trace(go.Scatter(
            x=df['date'],
            y=df['USDVND'],
            mode='lines',
            name='Tỷ giá USD/VND lịch sử',
            line=dict(color='blue')
        ))
        fig_forecast.add_trace(go.Scatter(
            x=forecasted_prices.index,
            y=forecasted_prices['Forecasted_USDVND'],
            mode='lines',
            name='Tỷ giá USD/VND dự báo',
            line=dict(color='red', dash='dash')
        ))
        fig_forecast.update_layout(
            title='Tỷ giá USD/VND lịch sử và dự báo',
            xaxis_title='Ngày',
            yaxis_title='Tỷ giá (VND)',
            height=400,
            margin=dict(l=40, r=20, t=40, b=40),  # Điều chỉnh lề
            xaxis=dict(showgrid=True, gridwidth=1, gridcolor='LightGrey'),
            yaxis=dict(showgrid=True, gridwidth=1, gridcolor='LightGrey')
        )
        st.plotly_chart(fig_forecast, use_container_width=True)