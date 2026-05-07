import streamlit as st
import pandas as pd
from datetime import datetime
from weather_agent import (
    build_agent_plan,
    build_summary_highlights,
    summarize_plan,
    summarize_weather_frame,
)
from weather_core import get_weather_data  # 你原来的函数保留在 wt_data.py

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

st.set_page_config(page_title="MeteoAgent", layout="centered")

st.title("🌦️ MeteoAgent")
st.caption("Task-oriented weather retrieval and analysis agent powered by NASA POWER.")

st.subheader("🤖 Weather Agent")
agent_prompt = st.text_area(
    "Describe the task in plain language",
    placeholder="Example: Summarize the weather for latitude 32 and longitude -84 from 2015-01-01 to 2015-12-31 in Celsius.",
    height=110,
)

if st.button("Run Agent Plan"):
    plan = build_agent_plan(agent_prompt)
    st.markdown("**Agent Plan**")
    st.json(summarize_plan(plan))
    if plan.next_actions:
        st.markdown("**Next Actions**")
        for action in plan.next_actions:
            st.write(f"- {action}")

    if plan.intent in {"fetch_weather", "summarize_weather"} and not plan.missing:
        with st.spinner("Agent is fetching weather data..."):
            try:
                df = get_weather_data(plan.lat, plan.lon, plan.start_date, plan.end_date, unit=plan.unit)
                st.success("✅ Agent completed the weather retrieval task.")
                st.dataframe(df)
                summary = summarize_weather_frame(df)
                highlights = build_summary_highlights(summary)
                if highlights:
                    st.markdown("**Agent Highlights**")
                    for line in highlights:
                        st.write(f"- {line}")
                with st.expander("Structured summary"):
                    st.json(summary)
                filename = f"agent_weather_{plan.start_date}_{plan.end_date}_{plan.lat}_{plan.lon}.csv"
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button("📥 Download Agent Result CSV", csv, file_name=filename, mime="text/csv")
            except Exception as e:
                st.error(f"❌ Agent failed: {e}")
    elif plan.intent == "compare_predictions":
        st.info("Upload the real and predicted CSV files below, then use the comparison section to finish the task.")
        if plan.compare_field:
            st.session_state["agent_compare_field"] = plan.compare_field
            st.success(f"Agent suggested comparison field: {plan.compare_field}")
    else:
        st.warning("The agent needs more structured details before it can execute the task.")

st.markdown("---")
st.subheader("Manual Weather Retrieval")

# --- Helper: parameter glossary ---
def render_param_glossary():
    with st.expander("ℹ️ Parameter glossary (click to expand)"):
        st.markdown(
            """
            **date** — Calendar date (YYYY/MM/DD in exports)

            **t_max** — Daily maximum 2 m air temperature.

            **t_min** — Daily minimum 2 m air temperature.

            **t_avg** — Daily mean 2 m air temperature.

            **t_range** — Daily temperature range (t_max − t_min).

            **precip** — Total daily precipitation.

            **solar_rad** — All-sky surface shortwave downwelling radiation (daily total at surface).

            **clrsky_solar_rad** — Clear-sky surface shortwave downwelling radiation (no cloud effect).

            **toa_solar_rad** — Top-of-atmosphere shortwave downwelling radiation (daily total above atmosphere).

            **rel_humidity** — 2 m relative humidity.

            **spec_humidity** — 2 m specific humidity.

            **wind_speed_10m** — Wind speed at 10 m above surface.

            **wind_speed_50m** — Wind speed at 50 m above surface.

            **wind_direction_10m** — Wind direction at 10 m (degrees from north, meteorological convention).

            **surface_pressure** — Surface air pressure.

            *Note:* Temperature columns (`t_*`) use the unit you selected above (Celsius/Kelvin). Other variables follow the original provider units from NASA POWER/Open-Meteo.
            """
        )

# 输入经纬度和日期范围
lat = st.number_input("Latitude", value=32.0, format="%.6f")
lon = st.number_input("Longitude", value=-84.0, format="%.6f")
start_date = st.date_input("Start Date", value=datetime(2015, 1, 1))
end_date = st.date_input("End Date", value=datetime(2015, 12, 31))

unit = st.radio("Temperature Unit", ["Celsius (°C)", "Kelvin (K)"])
unit_code = "C" if "Celsius" in unit else "K"

# Data source note
st.info("Data source: **NASA POWER** (Prediction Of Worldwide Energy Resources) — provides global meteorological and solar radiation data.")

# 按钮触发
if st.button("Get Weather Data"):
    if start_date > end_date:
        st.error("❌ Start date cannot be later than end date")
    else:
        with st.spinner("Fetching data, please wait..."):
            try:
                progress_bar = st.progress(0)
                progress_bar.progress(10)
                df = get_weather_data(lat, lon, str(start_date), str(end_date), unit=unit_code)
                progress_bar.progress(50)
                if df is not None and not df.empty:
                    progress_bar.progress(100)
                    st.success("✅ Success!")
                    st.dataframe(df)
                    render_param_glossary()

                    # 下载链接
                    filename = f"weather_{start_date}_{end_date}_{lat}_{lon}.csv"
                    csv = df.to_csv(index=False).encode("utf-8")
                    st.download_button("📥 Download CSV", csv, file_name=filename, mime="text/csv")
                else:
                    st.warning("⚠️ No valid data retrieved.")
            except Exception as e:
                st.error(f"❌ Error: {e}")

# --- 真实 vs 预测 CSV 数据对比模块 ---
st.markdown("---")
st.header("📊 Real vs Predicted Data Comparison")

# CSV format reminder (inline)
st.subheader("CSV format reminder")
st.info(
    """
    **Expected columns (example):** `t_max, t_min, t_avg, precip, solar_rad`  
    • First row must be headers, comma-separated  
    • Numeric values only (no units in cells)  
    • Optional `date` column allowed (used only for alignment/preview)  
    • The **Predicted CSV** must have exactly the same column names as the **Real Weather Data** output above (including any additional fields such as T2M_RANGE, RH2M, etc.) to ensure correct alignment and comparison  
    """
)

# Example small table
example_df = pd.DataFrame({
    "t_max": [298.51, 299.16, 295.12],
    "t_min": [287.43, 288.33, 282.39],
    "t_avg": [291.86, 292.42, 286.52],
    "precip": [0.04, 3.16, 15.19],
    "solar_rad": [3.5119, 1.1654, 0.9737]
})
st.caption("Example CSV (first rows)")
st.dataframe(example_df)

real_file = st.file_uploader("Upload Real Weather CSV", type=["csv"], key="real_file")
pred_file = st.file_uploader("Upload Predicted Weather CSV", type=["csv"], key="pred_file")

# 只有在 real_file 和 pred_file 都已上传时才进行分析
if real_file and pred_file:
    try:
        # 自动检测编码读取
        df_real = read_csv_with_encoding_detection(real_file)
        df_pred = read_csv_with_encoding_detection(pred_file)

        # 自动检测可对比的公共字段（包括温度、降水、光照等）
        compare_fields = [col for col in df_real.columns if col in df_pred.columns]
        # 移除无意义字段如日期、单位等非数值字段
        compare_fields = [col for col in compare_fields if col.lower() not in ['date', 'unit']]

        if not compare_fields:
            st.error("❌ No common fields found for comparison")
        else:
            default_index = 0
            suggested_field = st.session_state.get("agent_compare_field")
            if suggested_field in compare_fields:
                default_index = compare_fields.index(suggested_field)
            target_col = st.selectbox("Select field to compare:", compare_fields, index=default_index)
            if target_col.lower() == "date":
                st.warning("⚠️ The 'date' field is a time field; no error metrics or plots will be generated.")
            else:
                y_true = df_real[target_col].reset_index(drop=True)
                y_pred = df_pred[target_col].reset_index(drop=True)

                # 数据预览（前10行）
                st.subheader("📌 Preview (first 10 rows)")
                st.dataframe(pd.DataFrame({
                    "Actual": y_true.head(10),
                    "Predicted": y_pred.head(10)
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

                interval = st.selectbox("Select time scale", ["Weekly", "Biweekly", "Monthly"], key="interval_select")
                interval_en = interval

                if interval == "Weekly":
                    mae_vals = weekly_mae
                    rmse_vals = weekly_rmse
                    r2_vals = weekly_r2
                elif interval == "Biweekly":
                    mae_vals = biweekly_mae
                    rmse_vals = biweekly_rmse
                    r2_vals = biweekly_r2
                else:
                    mae_vals = monthly_mae
                    rmse_vals = monthly_rmse
                    r2_vals = monthly_r2

                with st.expander("📊 Detailed Error Metrics"):
                    st.subheader(f"{interval} MAE")
                    st.dataframe(pd.DataFrame({"Interval": list(range(1, len(mae_vals)+1)), "MAE": mae_vals}))

                    st.subheader(f"{interval} RMSE")
                    st.dataframe(pd.DataFrame({"Interval": list(range(1, len(rmse_vals)+1)), "RMSE": rmse_vals}))

                    st.subheader(f"{interval} R²")
                    st.dataframe(pd.DataFrame({"Interval": list(range(1, len(r2_vals)+1)), "R²": r2_vals}))

                    st.subheader("Top 10 Absolute Errors (AE)")
                    st.dataframe(ae.head(10))

                    st.subheader("Top 10 Errors (Error)")
                    st.dataframe(error.head(10))

                st.write(f"**MAE**: {mae:.3f}")
                st.write(f"**RMSE**: {rmse:.3f}")
                st.write(f"**R²**: {r2:.3f}")

                # 折线图
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(y_true.index, y_true, label="Actual")
                ax.plot(y_pred.index, y_pred, label="Predicted", linestyle="--")
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
        st.error(f"❌ Comparison error: {e}")
