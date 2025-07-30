import requests
import pandas as pd
from datetime import datetime, timedelta

# NOAA API 信息
TOKEN = 'ZCEDHhcPZGPuMapRSBGwHTkQSWhfQglH'  # 请替换为你自己的 NOAA API token
BASE_URL = 'https://www.ncei.noaa.gov/cdo-web/api/v2/data'

# 时间范围：2012 年下半年
start_date = datetime(2012, 7, 1)
end_date = datetime(2012, 12, 31)

# 格式化时间
start_str = start_date.strftime('%Y-%m-%d')
end_str = end_date.strftime('%Y-%m-%d')

# 请求参数（通用参数，具体 datatypeid 在后续指定）
# 'locationid': 'FIPS:13',  # FIPS 13 是佐治亚州
params = {
    'datasetid': 'GHCND',
    'stationid': 'GHCND:USW00013882',  # 示例站点：哥伦布机场
    'startdate': start_str,
    'enddate': end_str,
    'units': 'metric',
    'limit': 1000,
}

headers = {
    'token': TOKEN
}

# 分页请求与每日平均，增加自动重试功能
import time
from requests.exceptions import ReadTimeout, RequestException

limit = 1000
max_retries = 5

def fetch_data(datatypeid):
    results = []
    offset = 1
    while True:
        params_current = params.copy()
        params_current['datatypeid'] = datatypeid
        params_current.update({'limit': limit, 'offset': offset})
        retry_count = 0
        while retry_count < max_retries:
            try:
                response = requests.get(BASE_URL, headers=headers, params=params_current, timeout=30)
                if response.status_code == 200:
                    break
                elif response.status_code in [503, 429]:
                    print(f"{datatypeid} 遇到错误 {response.status_code}，重试 {retry_count + 1}")
                    time.sleep(5)
                    retry_count += 1
                else:
                    print(f"{datatypeid} 错误: {response.status_code}")
                    print(response.text)
                    return []
            except (ReadTimeout, RequestException) as e:
                print(f"{datatypeid} 网络异常: {e}")
                return []
        if response.status_code != 200:
            break
        data = response.json().get('results', [])
        if not data:
            break
        results.extend(data)
        offset += limit
    return results

def process_dataframe(data, label):
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    df_grouped = df.groupby('date')['value'].mean().reset_index()
    df_grouped.columns = ['Date', label]
    return df_grouped

results_max = fetch_data('TMAX')
results_min = fetch_data('TMIN')

if results_max:
    df_max_grouped = process_dataframe(results_max, 'Max_Temp_C')
    df_max_grouped.to_csv('georgia_daily_temp_max.csv', index=False, float_format="%.2f")
    print("CSV 文件已保存为 georgia_daily_temp_max.csv")
else:
    print("未返回任何最大温度数据。")

if results_min:
    df_min_grouped = process_dataframe(results_min, 'Min_Temp_C')
    df_min_grouped.to_csv('georgia_daily_temp_min.csv', index=False, float_format="%.2f")
    print("CSV 文件已保存为 georgia_daily_temp_min.csv")
else:
    print("未返回任何最小温度数据。")

if results_max and results_min:
    df_all = pd.merge(df_max_grouped, df_min_grouped, on='Date')
    df_all['Avg_Temp_C'] = ((df_all['Max_Temp_C'] + df_all['Min_Temp_C']) / 2).round(2)
    df_all.to_csv('georgia_daily_temp_avg.csv', index=False, float_format="%.2f")
    print("CSV 文件已保存为 georgia_daily_temp_avg.csv")
else:
    print("无法计算平均气温，因为最大温度或最小温度数据缺失。")