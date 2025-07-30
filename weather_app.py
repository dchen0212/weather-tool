import streamlit as st
import pandas as pd
from datetime import datetime
from weather_core import get_weather_data  # 你原来的函数保留在 wt_data.py 中

st.set_page_config(page_title="天气数据查询", layout="centered")

st.title("🌤️ 天气数据查询系统")

# 输入经纬度和日期范围
lat = st.number_input("纬度 (Latitude)", value=32.0, format="%.6f")
lon = st.number_input("经度 (Longitude)", value=-84.0, format="%.6f")
start_date = st.date_input("起始日期", value=datetime(2015, 1, 1))
end_date = st.date_input("结束日期", value=datetime(2015, 12, 31))

# 按钮触发
if st.button("获取天气数据"):
    if start_date > end_date:
        st.error("❌ 起始日期不能晚于结束日期")
    else:
        with st.spinner("正在获取数据，请稍候..."):
            try:
                df = get_weather_data(lat, lon, str(start_date), str(end_date))
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