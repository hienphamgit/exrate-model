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
df = get_data()

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
st.title("Mô hình dự báo tỷ giá hối đoái USD/VND")

st.markdown("## 1. Trực quan hoá dữ liệu")
st.markdown("Dữ liệu tỷ giá hối đoái USD/VND từ 2019 đến nay.")
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
# --- 1. Tạo Subplots Figure ---
# Tạo một figure với 1 dòng và 2 cột
fig = make_subplots(rows=1, cols=2,
                    subplot_titles=('Lợi suất Logarithmic của USD/VND theo thời gian', 'Phân phối lợi suất Logarithmic'),
                    horizontal_spacing=0.15) # Điều chỉnh khoảng cách ngang giữa các biểu đồ

# --- 2. Thêm Biểu đồ Lợi suất Logarithmic (trái) ---
fig.add_trace(
    go.Scatter(x=df['date'], y=df['log_return'], mode='lines', name='Lợi suất Log'),
    row=1, col=1 # Đặt vào hàng 1, cột 1
)
# --- 3. Thêm Biểu đồ Phân phối (Histogram + KDE) (phải) ---

# Histogram Trace cho biểu đồ phân phối
hist_trace = go.Histogram(
    x=df['log_return'],
    nbinsx=50,
    name='Lợi suất Log',
    marker_color='lightblue',
    opacity=0.8,
    histnorm='probability density', # Chuẩn hóa histogram về mật độ để khớp với KDE
    showlegend=False # Không hiển thị chú giải cho histogram trong subplot này
)
fig.add_trace(hist_trace, row=1, col=2) # Đặt vào hàng 1, cột 2

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
    showlegend=False # Không hiển thị chú giải cho KDE trong subplot này
)
fig.add_trace(kde_trace, row=1, col=2) # Đặt vào hàng 1, cột 2

# --- 4. Cập nhật Layout của Biểu đồ tổng thể ---
fig.update_layout(
    title_text='Phân tích Lợi suất Logarithmic',
    height=500, # Chiều cao tổng thể của figure
    # Cập nhật nhãn trục cho subplot đầu tiên (Lợi suất Logarithmic)
    xaxis=dict(title_text='', showgrid=True, gridwidth=1, gridcolor='LightGrey'),
    yaxis=dict(title_text='Lợi suất Logarithmic', showgrid=True, gridwidth=1, gridcolor='LightGrey'),
    # Cập nhật nhãn trục cho subplot thứ hai (Phân phối)
    xaxis2=dict(title_text='', showgrid=True, gridwidth=1, gridcolor='LightGrey'), # xaxis2 cho subplot thứ 2
    yaxis2=dict(title_text='', showgrid=True, gridwidth=1, gridcolor='LightGrey') # yaxis2 cho subplot thứ 2
)

# --- 5. Hiển thị biểu đồ trên Streamlit ---
st.plotly_chart(fig, use_container_width=True)

# Xây dựng mô hình dự báo tỷ giá hối đoái bằng Garch
st.markdown("## 3. Dự báo tỷ giá hối đoái USD/VND")

col1, col2 = st.columns(2)

with col1:
    # Widget chọn khoảng thời gian dự báo trong cột 1
    forecast_period = st.slider(
        "Chọn khoảng thời gian dự báo (ngày)",
        min_value=1,
        max_value=365,
        value=30
    )
    # Thêm radio button để chọn tìm mô hình hiệu quả nhất hoặc manually chọn mô hình. Nếu chọn manually thì sẽ hiện ra các lựa chọn mô hình GARCH
    model_type = st.radio(
        "Chọn mô hình GARCH",
        options=["Tự động tìm mô hình tốt nhất", "GARCH(1, 1)", "GARCH(1, 2)", "GARCH(2, 1)", "GARCH(2, 2)"],
        index=0,  # Mặc định chọn "Tự động tìm mô hình tốt nhất"
        horizontal=True
    )

# Xây dựng mô hình GARCH
if model_type == "Tự động tìm mô hình tốt nhất":
    # Tự động tìm mô hình GARCH tốt nhất bằng vòng lặp
    best_aic = np.inf
    best_model = None
    for p in range(1, 3):  # Thử p = 1
        for q in range(1, 3):
            try:
                model = arch_model(df['log_return'], vol='Garch', p=p, q=q, dist='t')
                model_fit = model.fit(disp="off")
                if model_fit.aic < best_aic:
                    best_aic = model_fit.aic
                    best_model = model_fit
            except Exception as e:
                st.error(f"Lỗi khi xây dựng mô hình GARCH({p}, {q}): {e}")
                continue
    if best_model is not None:
        st.success(f"Đã tìm thấy mô hình GARCH tốt nhất: {best_model.model.__class__.__name__} với AIC = {best_aic:.2f}")
        model_fit = best_model    
else:
    # Chọn mô hình GARCH theo lựa chọn của người dùng
    p, q = 1, 1
    if model_type == "GARCH(1, 2)":
        p, q = 1, 2
    elif model_type == "GARCH(2, 1)":
        p, q = 2, 1
    elif model_type == "GARCH(2, 2)":
        p, q = 2, 2
    model = arch_model(df['log_return'], vol='Garch', p=p, q=q)
    model_fit = model.fit(disp="off")
# Hiển thị thông tin mô hình vào cột 2
with col2:
    st.markdown("### Thông tin mô hình GARCH")
    st.write(f"Mô hình: {model_fit.model.__class__.__name__}")
    st.write(f"AIC: {model_fit.aic:.2f}")
    st.write(f"BIC: {model_fit.bic:.2f}")
    st.write(f"Log-Likelihood: {model_fit.loglikelihood:.2f}")
# Hiển thị tóm tắt mô hình
col3, col4 = st.columns(2)
with col3:
    st.markdown("### Tóm tắt mô hình GARCH")
    st.write(model_fit.summary())
with col4:
    st.markdown("### Biểu đồ phân tích mô hình GARCH")
    # Vẽ biểu đồ phân tích mô hình GARCH
    fig_garch = model_fit.plot(annualize='D')
    st.pyplot(fig_garch)

# Dự báo biến động trong tương lai
forecast = model_fit.forecast(horizon=forecast_period)
print(forecast.mean.tail())
# Hiển thị kết quả tỷ giá hối đoái dự báo
st.markdown("## 4. Dự báo tỷ giá hối đoái USD/VND ")
# Tạo DataFrame cho kết quả dự báo
forecast_df = pd.DataFrame({
    'date': pd.date_range(start=df['date'].iloc[-1] + pd.Timedelta(days=1), periods=forecast_period, freq='D'),
    'forecast': forecast.mean.iloc[-forecast_period:].values[0]
})  
forecast_df['forecast'] = np.exp(forecast_df['forecast']) * df['USDVND'].iloc[-1]  # Chuyển đổi từ log return về giá trị thực tế

#Chia làm 2 cột để hiển thị dự báo và tỷ giá thực tế
col5, col6 = st.columns(2)
with col5:
    st.dataframe(forecast_df, use_container_width=True)
with col6:
    #Cắt bớt dữ liệu df để chỉ lấy 1 năm gần nhất cho biểu đồ
    df_recent = df[df['date'] >= (df['date'].max() - pd.Timedelta(days=365))]
    # Hiển thị biểu đồ dự báo tỷ giá hối đoái
    fig_forecast = go.Figure()
    fig_forecast.add_trace(go.Scatter(x=df_recent['date'], y=df_recent['USDVND'], mode='lines', name='USD/VND (Thực tế)'))
    fig_forecast.add_trace(go.Scatter(x=forecast_df['date'], y=forecast_df['forecast'], mode='lines', name='USD/VND (Dự báo)', line=dict(dash='dash')))
    fig_forecast.update_layout(title='Dự báo tỷ giá hối đoái USD/VND', xaxis_title='Ngày', yaxis_title='Tỷ giá (VND)')
    st.plotly_chart(fig_forecast, use_container_width=True)
