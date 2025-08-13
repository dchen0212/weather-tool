import requests
import pandas as pd
import chardet

# -------------------- CSV 编码检测与读取 --------------------
def read_csv_with_encoding_detection(file_obj):
    """自动检测编码并读取 CSV 文件"""
    raw = file_obj.read()
    result = chardet.detect(raw)
    encoding = result["encoding"] or "utf-8"
    file_obj.seek(0)  # 重置文件指针
    try:
        df = pd.read_csv(file_obj, encoding=encoding)
        return df
    except Exception as e:
        raise ValueError(f"读取失败，尝试使用编码 {encoding}，错误：{e}")

# -------------------- NASA POWER 参数列表获取 --------------------
def _get_nasa_daily_parameter_list(community="RE"):
    """从 NASA POWER 元数据端点获取 *全部* 日尺度参数 ID 列表。
    成功时返回如 ["T2M", "RH2M", ...] 的列表；失败时回退到一个安全子集。
    """
    try:
        meta_url_candidates = [
            "https://power.larc.nasa.gov/api/parameters/temporal/daily",
            "https://power.larc.nasa.gov/api/v1/parameters/temporal/daily",
        ]
        for url in meta_url_candidates:
            try:
                resp = requests.get(url, params={"community": community, "format": "JSON"}, timeout=20)
                data = resp.json()
                if isinstance(data, dict):
                    # 兼容两种返回结构
                    if "parameters" in data and isinstance(data["parameters"], dict):
                        return sorted(list(data["parameters"].keys()))
                    if "properties" in data and "parameter" in data["properties"] and isinstance(data["properties"]["parameter"], dict):
                        return sorted(list(data["properties"]["parameter"].keys()))
            except Exception:
                continue
    except Exception:
        pass
    # 失败回退
    return [
        "T2M", "T2M_MAX", "T2M_MIN", "T2M_RANGE",
        "RH2M", "QV2M", "WS10M", "WS50M", "WD10M",
        "PRECTOTCORR", "PRECTOT", "SNOWC", "PS",
        "ALLSKY_SFC_SW_DWN", "CLRSKY_SFC_SW_DWN", "TOA_SW_DWN"
    ]

# -------------------- API 3: NASA POWER --------------------
def get_weather_nasa_power(lat, lon, start_date, end_date, unit="C"):
    try:
        print("尝试使用 NASA POWER API...")
        start_fmt = start_date.replace("-", "")
        end_fmt = end_date.replace("-", "")
        url = "https://power.larc.nasa.gov/api/temporal/daily/point"

        # 我们同时尝试多个社区以拿到更全的日尺度参数
        communities = ["RE", "AG", "SB"]

        def _chunks(seq, size):
            for i in range(0, len(seq), size):
                yield seq[i:i+size]

        df_final = None
        for comm in communities:
            all_params = _get_nasa_daily_parameter_list(community=comm)
            if not all_params:
                continue

            df_comm = None
            for batch in _chunks(all_params, 35):  # 控制每批参数数量，避免 URL 过长
                params_str = ",".join(batch)
                q = {
                    "start": start_fmt,
                    "end": end_fmt,
                    "latitude": lat,
                    "longitude": lon,
                    "parameters": params_str,
                    "format": "JSON",
                    "community": comm
                }
                resp = requests.get(url, params=q, timeout=20)
                print(f"POWER 社区 {comm} 批次响应预览：", resp.text[:400])
                data = resp.json()
                if "properties" not in data or "parameter" not in data["properties"]:
                    continue
                pmap = data["properties"]["parameter"]  # {PARAM: {date: value}}

                # 收集本批所有日期
                dates = set()
                for series in pmap.values():
                    if isinstance(series, dict):
                        dates.update(series.keys())
                if not dates:
                    continue
                dates = sorted(dates)

                # 构建本批 DataFrame
                rows = []
                for d in dates:
                    row = {"date": d}
                    for p in pmap.keys():
                        v = pmap.get(p, {}).get(d, None)
                        row[p] = v
                    rows.append(row)
                df_batch = pd.DataFrame(rows)

                if df_comm is None:
                    df_comm = df_batch
                else:
                    # 合并到该社区的累计表
                    df_comm = pd.merge(df_comm, df_batch, on="date", how="outer")

            if df_comm is None:
                continue

            # 与总表合并，去除重复列（保留已有列）
            if df_final is None:
                df_final = df_comm
            else:
                dup_cols = list(set(df_comm.columns) & set(df_final.columns) - {"date"})
                if dup_cols:
                    df_comm = df_comm.drop(columns=dup_cols)
                df_final = pd.merge(df_final, df_comm, on="date", how="outer")

        if df_final is None or df_final.empty:
            return None

        # 单位处理：如需 Kelvin，把所有 T2M* 字段 C->K
        if unit.upper() == "K":
            for col in list(df_final.columns):
                if col.upper().startswith("T2M"):
                    with pd.option_context('mode.use_inf_as_na', True):
                        df_final[col] = pd.to_numeric(df_final[col], errors='coerce') + 273.15
            df_final["unit"] = "K"
        else:
            df_final["unit"] = "C"

        # 排序列：date 在前
        other_cols = [c for c in df_final.columns if c != "date"]
        df_final = df_final[["date"] + other_cols]
        return df_final
    except Exception as e:
        print(f"NASA POWER 错误: {e}")
        return None

# -------------------- API: Open-Meteo --------------------
def get_weather_open_meteo(lat, lon, start_date, end_date):
    try:
        print("尝试使用 Open-Meteo API...")
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "daily": "temperature_2m_max,temperature_2m_min,temperature_2m_mean,precipitation_sum,shortwave_radiation_sum",
            "timezone": "auto"
        }
        resp = requests.get(url, params=params, timeout=10)
        print("Open-Meteo 返回原始响应：", resp.text)
        print("HTTP 状态码：", resp.status_code)
        data = resp.json()
        if "daily" not in data:
            return None
        df = pd.DataFrame({
            "date": data["daily"]["time"],
            "t_max": data["daily"]["temperature_2m_max"],
            "t_min": data["daily"]["temperature_2m_min"],
            "t_avg": data["daily"]["temperature_2m_mean"],
            "precip": data["daily"]["precipitation_sum"],
            "solar_rad": data["daily"]["shortwave_radiation_sum"]
        })
        df = df[["date", "t_max", "t_min", "t_avg", "precip", "solar_rad"]]
        return df
    except Exception as e:
        print(f"Open-Meteo 错误: {e}")
        return None

# -------------------- 自动切换 API --------------------
def get_weather_data(lat, lon, start_date, end_date, unit="C"):
    apis = [
        get_weather_nasa_power,
        get_weather_open_meteo,
    ]
    for api_func in apis:
        if api_func == get_weather_nasa_power:
            df = api_func(lat, lon, start_date, end_date, unit)
        else:
            df = api_func(lat, lon, start_date, end_date)
        if df is not None and not df.empty:
            print(f"✅ 成功使用 {api_func.__name__} 数据源")
            return df

    raise Exception("❌ 所有数据源都无法获取数据")



# -------------------- 评估与绘图工具函数 --------------------
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_and_plot_predictions(y_true, y_pred, field_name):
    """裁剪长度一致，计算 MAE/RMSE/R² 并画拟合图"""
    import streamlit as st
    if field_name.lower() == "date":
        st.warning("⚠️ 字段 'date' 为时间字段，不进行误差计算。")
        return
    min_len = min(len(y_true), len(y_pred))
    y_true = y_true[:min_len]
    y_pred = y_pred[:min_len]

    # 计算误差指标
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)

    # 显示指标
    st.markdown(f"**MAE**: {mae:.3f}")
    st.markdown(f"**RMSE**: {rmse:.3f}")
    st.markdown(f"**R²**: {r2:.3f}")

    # 折线图（原来的）
    fig1, ax1 = plt.subplots()
    ax1.plot(getattr(y_true, "values", y_true), label="真实值", linewidth=1.5)
    ax1.plot(getattr(y_pred, "values", y_pred), "--", label="预测值", linewidth=1.5)
    ax1.set_title(f"{field_name} 时间序列对比")
    ax1.legend()
    st.pyplot(fig1)

    # 拟合图：真实 vs 预测
    fig2, ax2 = plt.subplots()
    ax2.scatter(y_true, y_pred, s=10, alpha=0.6)
    z = np.polyfit(y_true, y_pred, 1)
    p = np.poly1d(z)
    ax2.plot(y_true, p(y_true), "r--", label="拟合线")
    ax2.set_xlabel("真实值")
    ax2.set_ylabel("预测值")
    ax2.set_title(f"{field_name} 拟合图")
    ax2.legend()
    st.pyplot(fig2)