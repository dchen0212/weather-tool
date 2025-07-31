import streamlit as st
import pandas as pd
from datetime import datetime
from weather_core import get_weather_data  # 你原来的函数保留在 wt_data.py

# 自动检测编码读取 CSV 文件
def read_csv_with_encoding_detection(uploaded_file):
    import chardet
    raw = uploaded_file.read()
    result = chardet.detect(raw)
    encoding = result['encoding']
    uploaded_file.seek(0)
    return pd.read_csv(uploaded_file, encoding=encoding)

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


# --- 真实 vs 预测数据对比分析 ---
st.markdown("---")
st.header("📊 真实 vs 预测数据对比分析")

real_file = st.file_uploader("📂 上传真实天气 CSV 文件", type="csv", key="real")
pred_file = st.file_uploader("📂 上传预测天气 CSV 文件", type="csv", key="pred")

if real_file and pred_file:
    try:
        # 自动检测编码读取
        df_real = read_csv_with_encoding_detection(real_file)
        df_pred = read_csv_with_encoding_detection(pred_file)

        # 按列名自动检测公共字段
        common_cols = [col for col in df_real.columns if col in df_pred.columns]
        # 移除无意义字段如日期字段
        common_cols = [col for col in common_cols if col.lower() != 'date']
        if not common_cols:
            st.error("❌ 未找到两个文件中共有的对比字段")
        else:
            target_col = st.selectbox("请选择对比字段：", common_cols)
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

                st.write(f"**MAE**: {mae:.3f}")
                st.write(f"**RMSE**: {rmse:.3f}")
                st.write(f"**R²**: {r2:.3f}")

                ae = np.abs(y_true - y_pred)
                error = y_pred - y_true
                with st.expander("查看前10个绝对误差 (AE)"):
                    st.dataframe(ae.head(10))

                with st.expander("查看前10个误差 (Error)"):
                    st.dataframe(error.head(10))

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
    except Exception as e:
        st.error(f"❌ 对比出错：{e}")
