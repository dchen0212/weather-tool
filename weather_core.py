import requests
import pandas as pd

# -------------------- API 3: NASA POWER --------------------
def get_weather_nasa_power(lat, lon, start_date, end_date, unit="C"):
    try:
        print("尝试使用 NASA POWER API...")
        start_fmt = start_date.replace("-", "")
        end_fmt = end_date.replace("-", "")
        url = "https://power.larc.nasa.gov/api/temporal/daily/point"
        params = {
            "start": start_fmt,
            "end": end_fmt,
            "latitude": lat,
            "longitude": lon,
            "parameters": "T2M,T2M_MAX,T2M_MIN,PRECTOTCORR,ALLSKY_SFC_SW_DWN",
            "format": "JSON",
            "community": "RE"
        }
        resp = requests.get(url, params=params, timeout=10)
        print("NASA POWER 原始响应：", resp.text)
        print("HTTP 状态码：", resp.status_code)
        data = resp.json()
        if "properties" not in data or "parameter" not in data["properties"]:
            print("NASA POWER 返回数据格式异常")
            return None
        t_avg = data["properties"]["parameter"].get("T2M", {})
        t_max = data["properties"]["parameter"].get("T2M_MAX", {})
        t_min = data["properties"]["parameter"].get("T2M_MIN", {})

        records = []
        for date in t_avg:
            records.append({
                "date": date,
                "t_max": t_max.get(date),
                "t_min": t_min.get(date),
                "t_avg": t_avg.get(date),
                "precip": data["properties"]["parameter"].get("PRECTOTCORR", {}).get(date),
                "solar_rad": data["properties"]["parameter"].get("ALLSKY_SFC_SW_DWN", {}).get(date),
            })
        df = pd.DataFrame(records)
        df = df[["date", "t_max", "t_min", "t_avg", "precip", "solar_rad"]]
        if unit.upper() == "K":
            df["t_max"] = df["t_max"] + 273.15
            df["t_min"] = df["t_min"] + 273.15
            df["t_avg"] = df["t_avg"] + 273.15
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
