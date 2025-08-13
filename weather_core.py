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
    """获取 NASA POWER 可用的每日参数列表"""
    metadata_urls = [
        "https://power.larc.nasa.gov/api/metadata/parameter-names",
        "https://power.larc.nasa.gov/api/temporal/daily/point?community={}&parameters=ALL&format=JSON".format(community),
    ]
    for url in metadata_urls:
        try:
            resp = requests.get(url, timeout=20)
            if resp.status_code == 200:
                data = resp.json()
                # 尝试解析参数列表
                if "parameters" in data:
                    # 可能是参数元数据接口
                    params = list(data["parameters"].keys())
                    if params:
                        return params
                elif "properties" in data and "parameter" in data["properties"]:
                    # 可能是示例数据接口，取参数键
                    params = list(data["properties"]["parameter"].keys())
                    if params:
                        return params
        except Exception:
            continue
    # 如果都失败，返回一个安全的最小参数集
    return [
        "T2M", "T2M_MAX", "T2M_MIN",
        "PRECTOT", "ALLSKY_SFC_SW_DWN"
    ]

# -------------------- API 3: NASA POWER --------------------
def get_weather_nasa_power(lat, lon, start_date, end_date, unit="C"):
    try:
        print("尝试使用 NASA POWER API...")
        start_fmt = start_date.replace("-", "")
        end_fmt = end_date.replace("-", "")
        params_list = _get_nasa_daily_parameter_list()
        # 参数字符串
        parameters = ",".join(params_list)
        url = "https://power.larc.nasa.gov/api/temporal/daily/point"
        params = {
            "start": start_fmt,
            "end": end_fmt,
            "latitude": lat,
            "longitude": lon,
            "parameters": parameters,
            "format": "JSON",
            "community": "RE"
        }
        resp = requests.get(url, params=params, timeout=20)
        print("NASA POWER 原始响应前500字符：", resp.text[:500])
        print("HTTP 状态码：", resp.status_code)
        data = resp.json()
        if "properties" not in data or "parameter" not in data["properties"]:
            print("NASA POWER 返回数据格式异常")
            return None

        parameters = data["properties"]["parameter"]
        # Use dates from one of the parameters as base
        any_param = next(iter(parameters.values()))
        records = []
        for date in any_param:
            record = {"date": date}
            for param_key, param_values in parameters.items():
                record[param_key.lower()] = param_values.get(date)
            records.append(record)

        df = pd.DataFrame(records)
        # Reorder columns: date first, then all parameters
        cols = ["date"] + [c for c in df.columns if c != "date"]
        df = df[cols]

        if unit.upper() == "K":
            for temp_key in ["t2m_max", "t2m_min", "t2m"]:
                if temp_key in df.columns:
                    df[temp_key] = df[temp_key] + 273.15
            df["unit"] = "K"
        else:
            df["unit"] = "C"
        return df
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