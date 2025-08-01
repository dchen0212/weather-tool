import streamlit as st
import pandas as pd
from datetime import datetime
from weather_core import get_weather_data  # 你原来的函数保留在 wt_data.py
import io
import tempfile
import os

# 自动检测编码读取 CSV 文件
def read_csv_with_encoding_detection(uploaded_file):
    import chardet
    pos = uploaded_file.tell()
    sample = uploaded_file.read(1024)
    result = chardet.detect(sample)
    encoding = result['encoding']
    uploaded_file.seek(pos)
    df = pd.read_csv(uploaded_file, encoding=encoding)
    uploaded_file.seek(0)
    return df

st.set_page_config(page_title="天气数据查询", layout="centered")

st.title("🌤️ 天气数据查询系统")

# 输入经纬度和日期范围
lat = st.number_input("纬度 (Latitude)", value=32.0, format="%.6f")
lon = st.number_input("经度 (Longitude)", value=-84.0, format="%.6f")
start_date = st.date_input("起始日期", value=datetime(2015, 1, 1))
end_date = st.date_input("结束日期", value=datetime(2015, 12, 31))

unit = st.radio("温度单位", ["摄氏度 (°C)", "开尔文 (K)"])
unit_code = "C" if "摄氏" in unit else "K"

# 按钮触发
if st.button("获取天气数据"):
    if start_date > end_date:
        st.error("❌ 起始日期不能晚于结束日期")
    else:
        with st.spinner("正在获取数据，请稍候..."):
            try:
                df = get_weather_data(lat, lon, str(start_date), str(end_date), unit=unit_code)
                if df is not None and not df.empty:
                    st.success("✅ 获取成功！")
                    st.dataframe(df)

                    # 下载链接
                    filename = f"weather_{start_date}_{end_date}_{lat}_{lon}.csv"
                    csv = df.to_csv(index=False).encode("utf-8")
                    st.download_button("📥 下载 CSV 文件", csv, file_name=filename, mime="text/csv")
                else:
                    st.warning("⚠️ 没有获取到有效数据。")
            except Exception as e:
                st.error(f"❌ 出错：{e}")

# --- 预测 NC 文件读取与 CSV 导出模块 ---
st.markdown("---")
st.header("📂 预测 NC 数据转换为 CSV")

import netCDF4 as nc
import numpy as np
import h5py

# 天气参数分类关键词
weather_categories = {
    'temperature': {'keywords': ['temp', 't2m', 'temperature', 'air_temp', 'ta']},
    'wind': {'keywords': ['wind', 'u', 'v', 'wind_speed', 'ua', 'va']},
    'humidity': {'keywords': ['humidity', 'rh', 'q', 'hus']},
    'pressure': {'keywords': ['pressure', 'sp', 'slp']},
    'precipitation': {'keywords': ['precip', 'rain', 'snow', 'prcp']},
    'radiation': {'keywords': ['rad', 'solar', 'swdown']},
    'geopotential': {'keywords': ['zg', 'geopotential', 'height']}
}

def identify_weather_vars(nc_file):
    """识别文件中的天气变量并分类"""
    identified = {}
    for var_name in nc_file.variables:
        if var_name.lower() in ['time', 'latitude', 'longitude', 'lat', 'lon', 'level', 'pressure']:
            continue
        var_name_lower = var_name.lower()
        for category, props in weather_categories.items():
            if any(kw in var_name_lower for kw in props['keywords']):
                identified[var_name] = category
                break
    return identified

def extract_location_data(var, lat_idx, lon_idx):
    """提取特定位置的数据并展平"""
    try:
        dims = var.dimensions
        if len(dims) == 4:
            data = var[:, 0, lat_idx, lon_idx]
        elif len(dims) == 3:
            data = var[:, lat_idx, lon_idx]
        elif len(dims) == 2:
            data = var[:, lat_idx]
        else:
            data = var[:]
        if hasattr(data, "ndim"):
            if data.ndim == 0:
                data = [data.item()]
            elif data.ndim == 1:
                data = data
            else:
                data = data.flatten()
        return data
    except Exception:
        return None

def process_nc_streamlit(uploaded_file):
    """处理上传的NC文件"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".nc") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file.flush()
            try:
                ds = nc.Dataset(tmp_file.name, 'r')
                result = process_valid_nc(ds)
                ds.close()
                os.unlink(tmp_file.name)
                if result is None:
                    return pd.DataFrame()
                return result
            except OSError as e:
                if 'NetCDF: HDF error' in str(e):
                    try:
                        with h5py.File(tmp_file.name, 'r') as h5_file:
                            os.unlink(tmp_file.name)
                            st.error("⚠️ 暂不支持复杂HDF5解析，这里可扩展")
                            return pd.DataFrame()
                    except Exception as e2:
                        os.unlink(tmp_file.name)
                        st.error(f"❌ HDF5处理失败: {e2}")
                        return pd.DataFrame()
                else:
                    os.unlink(tmp_file.name)
                    st.error(f"❌ 处理失败: {e}")
                    return pd.DataFrame()
    except Exception as e:
        st.error(f"❌ 处理失败: {e}")
        return pd.DataFrame()

def process_valid_nc(nc_file):
    """正常netCDF4处理"""
    lat_var = nc_file.variables.get('latitude') or nc_file.variables.get('lat')
    lon_var = nc_file.variables.get('longitude') or nc_file.variables.get('lon')
    if lat_var is None or lon_var is None:
        st.error("❌ 缺少经纬度变量")
        return None
    lats = lat_var[:]
    lons = lon_var[:]
    lat_idx = np.abs(lats - lats.mean()).argmin()
    lon_idx = np.abs(lons - lons.mean()).argmin()
    actual_lat, actual_lon = lats[lat_idx], lons[lon_idx]

    time_var = nc_file.variables.get('time')
    if time_var is not None:
        try:
            times = nc.num2date(time_var[:], time_var.units)
            time_strs = [t.strftime('%Y-%m-%d %H:%M:%S') for t in times]
        except Exception:
            time_strs = [f"time_{i}" for i in range(len(time_var))]
    else:
        time_strs = [f"time_{i}" for i in range(10)]

    data = {
        "date": time_strs,
        "lat": [actual_lat] * len(time_strs),
        "lon": [actual_lon] * len(time_strs)
    }

    weather_vars = identify_weather_vars(nc_file)
    for var_name, category in weather_vars.items():
        var = nc_file.variables[var_name]
        var_data = extract_location_data(var, lat_idx, lon_idx)
        if var_data is not None and len(var_data) == len(time_strs):
            data[f"{category}_{var_name}"] = var_data

    return pd.DataFrame(data)

# 文件上传
nc_file = st.file_uploader("上传预测 NC 文件（.nc）", type=["nc"], key="pred_nc")
if nc_file is not None:
    df_nc = process_nc_streamlit(nc_file)
    if df_nc is not None and not df_nc.empty:
        st.write(f"**纬度 (Latitude)**: {df_nc['lat'].iloc[0]}")
        st.write(f"**经度 (Longitude)**: {df_nc['lon'].iloc[0]}")
        st.subheader("📌 预测 NC 数据预览")
        st.dataframe(df_nc.head(10))
        csv_data = df_nc.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 下载预测数据 CSV",
            csv_data,
            file_name="predicted_nc_data.csv",
            mime="text/csv"
        )

# --- 真实 vs 预测 CSV 数据对比模块 ---
real_file = st.file_uploader("上传真实天气 CSV 文件", type=["csv"], key="real_file")
pred_file = st.file_uploader("上传预测天气 CSV 文件", type=["csv"], key="pred_file")

# 只有在 real_file 和 pred_file 都已上传时才进行分析
if real_file and pred_file:
    st.header("📊 真实 vs 预测数据对比分析")
    try:
        # 自动检测编码读取
        df_real = read_csv_with_encoding_detection(real_file)
        df_pred = read_csv_with_encoding_detection(pred_file)

        # 自动检测可对比的公共字段（包括温度、降水、光照等）
        compare_fields = [col for col in df_real.columns if col in df_pred.columns]
        # 移除无意义字段如日期字段
        compare_fields = [col for col in compare_fields if col.lower() != 'date']

        if not compare_fields:
            st.error("❌ 未找到两个文件中共有的对比字段")
        else:
            target_col = st.selectbox("请选择对比字段：", compare_fields)
            if target_col.lower() == "date":
                st.warning("⚠️ 字段 'date' 为时间字段，不进行误差计算与绘图。")
            else:
                y_true = df_real[target_col].reset_index(drop=True)
                y_pred = df_pred[target_col].reset_index(drop=True)

                # 数据预览（前10行）
                st.subheader("📌 数据预览 (前10行)")
                st.dataframe(pd.DataFrame({
                    "真实值": y_true.head(10),
                    "预测值": y_pred.head(10)
                }))

                from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
                import numpy as np
                import matplotlib.pyplot as plt

                mae = mean_absolute_error(y_true, y_pred)
                rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                r2 = r2_score(y_true, y_pred)

                # 每7天计算 MAE 并绘图
                weekly_mae = [mean_absolute_error(y_true[i:i+7], y_pred[i:i+7]) for i in range(0, len(y_true), 7)]
                weekly_rmse = [np.sqrt(mean_squared_error(y_true[i:i+7], y_pred[i:i+7])) for i in range(0, len(y_true), 7)]
                weekly_r2 = [r2_score(y_true[i:i+7], y_pred[i:i+7]) for i in range(0, len(y_true), 7)]

                # 新增每两周和每月误差计算
                biweekly_mae = [mean_absolute_error(y_true[i:i+14], y_pred[i:i+14]) for i in range(0, len(y_true), 14)]
                monthly_mae = [mean_absolute_error(y_true[i:i+30], y_pred[i:i+30]) for i in range(0, len(y_true), 30)]

                biweekly_rmse = [np.sqrt(mean_squared_error(y_true[i:i+14], y_pred[i:i+14])) for i in range(0, len(y_true), 14)]
                monthly_rmse = [np.sqrt(mean_squared_error(y_true[i:i+30], y_pred[i:i+30])) for i in range(0, len(y_true), 30)]

                biweekly_r2 = [r2_score(y_true[i:i+14], y_pred[i:i+14]) for i in range(0, len(y_true), 14)]
                monthly_r2 = [r2_score(y_true[i:i+30], y_pred[i:i+30]) for i in range(0, len(y_true), 30)]

                ae = np.abs(y_true - y_pred)
                error = y_pred - y_true

                interval = st.selectbox("选择时间尺度", ["每周", "每两周", "每月"], key="interval_select")

                interval_map = {
                    "每周": "Weekly",
                    "每两周": "Biweekly",
                    "每月": "Monthly"
                }
                interval_en = interval_map[interval]

                if interval == "每周":
                    mae_vals = weekly_mae
                    rmse_vals = weekly_rmse
                    r2_vals = weekly_r2
                elif interval == "每两周":
                    mae_vals = biweekly_mae
                    rmse_vals = biweekly_rmse
                    r2_vals = biweekly_r2
                else:
                    mae_vals = monthly_mae
                    rmse_vals = monthly_rmse
                    r2_vals = monthly_r2

                with st.expander("📊 查看详细误差信息"):
                    st.subheader(f"{interval} MAE")
                    st.dataframe(pd.DataFrame({"Interval": list(range(1, len(mae_vals)+1)), "MAE": mae_vals}))

                    st.subheader(f"{interval} RMSE")
                    st.dataframe(pd.DataFrame({"Interval": list(range(1, len(rmse_vals)+1)), "RMSE": rmse_vals}))

                    st.subheader(f"{interval} R²")
                    st.dataframe(pd.DataFrame({"Interval": list(range(1, len(r2_vals)+1)), "R²": r2_vals}))

                    st.subheader("前10个绝对误差 (AE)")
                    st.dataframe(ae.head(10))

                    st.subheader("前10个误差 (Error)")
                    st.dataframe(error.head(10))

                st.write(f"**MAE**: {mae:.3f}")
                st.write(f"**RMSE**: {rmse:.3f}")
                st.write(f"**R²**: {r2:.3f}")

                # 折线图
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(y_true.index, y_true, label="真实值")
                ax.plot(y_pred.index, y_pred, label="预测值", linestyle="--")
                ax.set_title(f"{target_col} Comparison Line Chart")
                ax.set_xlabel("Index")
                ax.set_ylabel(target_col)
                ax.legend(["True Value", "Predicted Value"])
                st.pyplot(fig)

                fig2, ax2 = plt.subplots(figsize=(10, 4))
                ax2.plot(ae, label="Absolute Error (AE)")
                ax2.plot(error, label="Error")
                ax2.set_title(f"{target_col} Error Line Chart")
                ax2.set_xlabel("Index")
                ax2.set_ylabel("Error Value")
                ax2.legend()
                st.pyplot(fig2)

                fig3, ax3 = plt.subplots(figsize=(10, 4))
                ax3.plot(mae_vals, marker='o', label=f"{interval_en} MAE")
                ax3.set_title(f"{interval_en} MAE for {target_col}")
                ax3.set_xlabel("Interval Index")
                ax3.set_ylabel("MAE")
                ax3.legend()
                st.pyplot(fig3)

                fig4, ax4 = plt.subplots(figsize=(10, 4))
                ax4.plot(rmse_vals, marker='o', label=f"{interval_en} RMSE", color='orange')
                ax4.set_title(f"{interval_en} RMSE for {target_col}")
                ax4.set_xlabel("Interval Index")
                ax4.set_ylabel("RMSE")
                ax4.legend()
                st.pyplot(fig4)

                fig5, ax5 = plt.subplots(figsize=(10, 4))
                ax5.plot(r2_vals, marker='o', label=f"{interval_en} R²", color='green')
                ax5.set_title(f"{interval_en} R² for {target_col}")
                ax5.set_xlabel("Interval Index")
                ax5.set_ylabel("R²")
                ax5.legend()
                st.pyplot(fig5)
    except Exception as e:
        st.error(f"❌ 对比出错：{e}")
