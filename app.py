

import argparse
import pandas as pd
from weather_core import get_weather_data, standardize_weather_columns

def main():
    p = argparse.ArgumentParser(description="Fetch NASA POWER daily weather and save as CSV")
    p.add_argument("--lat", type=float, required=True, help="Latitude")
    p.add_argument("--lon", type=float, required=True, help="Longitude")
    p.add_argument("--start", type=str, required=True, help="YYYY-MM-DD")
    p.add_argument("--end", type=str, required=True, help="YYYY-MM-DD")
    p.add_argument("--unit", type=str, default="C", choices=["C", "K"])
    p.add_argument("--out", type=str, default="weather.csv")
    args = p.parse_args()

    df = get_weather_data(args.lat, args.lon, args.start, args.end, args.unit)
    if df is None or df.empty:
        raise SystemExit("No data returned from NASA POWER.")
    df = standardize_weather_columns(df)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df.to_csv(args.out, index=False)
    print(f"Saved: {args.out}  rows={len(df)}")

if __name__ == "__main__":
    main()