import sys
from weather_core import get_weather_data

print("📍 请输入所需参数：")
try:
    lat = float(input("纬度（latitude）："))
    lon = float(input("经度（longitude）："))
except ValueError:
    print("❌ 经纬度必须是数字，请重新运行程序。")
    sys.exit(1)

try:
    start_date = input("起始日期（YYYY-MM-DD）：")
    end_date = input("结束日期（YYYY-MM-DD）：")
    # 检查日期格式
    from datetime import datetime
    datetime.strptime(start_date, "%Y-%m-%d")
    datetime.strptime(end_date, "%Y-%m-%d")
except ValueError:
    print("❌ 日期格式错误，请使用 YYYY-MM-DD 格式。")
    sys.exit(1)

try:
    df = get_weather_data(lat, lon, start_date, end_date)
    print(df)
    output_file = f"weather_{start_date}_{end_date}_{lat}_{lon}.csv"
    df.to_csv(output_file, index=False)
    print(f"✅ 数据已保存为 {output_file}")
except Exception as e:
    print(f"❌ 出现错误：{e}")
