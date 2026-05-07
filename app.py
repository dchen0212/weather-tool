import argparse
import os


def launch_streamlit():
    from streamlit.web.bootstrap import run

    script = os.path.join(os.path.dirname(__file__), "weather_app.py")
    run(script, "", [], {})


def run_cli(args):
    import pandas as pd

    from weather_agent import build_agent_plan
    from weather_core import get_weather_data, standardize_weather_columns

    if args.prompt:
        plan = build_agent_plan(args.prompt)
        if plan.intent != "fetch_weather":
            raise SystemExit("Prompt mode in CLI currently supports weather retrieval requests only.")
        if plan.missing:
            raise SystemExit(f"Prompt is missing required fields: {', '.join(plan.missing)}")
        args.lat = plan.lat
        args.lon = plan.lon
        args.start = plan.start_date
        args.end = plan.end_date
        args.unit = plan.unit

    df = get_weather_data(args.lat, args.lon, args.start, args.end, args.unit)
    if df is None or df.empty:
        raise SystemExit("No data returned from NASA POWER.")

    df = standardize_weather_columns(df)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")

    df.to_csv(args.out, index=False)
    print(f"Saved: {args.out}  rows={len(df)}")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Weather Tool: launch the Streamlit UI or fetch NASA POWER weather data as CSV."
    )
    parser.add_argument("--gui", action="store_true", help="Launch the Streamlit interface explicitly.")
    parser.add_argument("--lat", type=float, help="Latitude")
    parser.add_argument("--lon", type=float, help="Longitude")
    parser.add_argument("--start", type=str, help="Start date in YYYY-MM-DD")
    parser.add_argument("--end", type=str, help="End date in YYYY-MM-DD")
    parser.add_argument("--unit", type=str, default="C", choices=["C", "K"], help="Temperature unit")
    parser.add_argument("--out", type=str, default="weather.csv", help="Output CSV filename")
    parser.add_argument("--prompt", type=str, help="Natural-language weather task, for example: fetch weather for lat 32 lon -84 from 2015-01-01 to 2015-12-31")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    cli_fields = [args.lat, args.lon, args.start, args.end, args.prompt]
    has_cli_args = any(value is not None for value in cli_fields)

    if args.gui or not has_cli_args:
        launch_streamlit()
        return

    missing = []
    if args.prompt:
        run_cli(args)
        return

    if args.lat is None:
        missing.append("--lat")
    if args.lon is None:
        missing.append("--lon")
    if args.start is None:
        missing.append("--start")
    if args.end is None:
        missing.append("--end")

    if missing:
        parser.error(f"CLI mode requires: {', '.join(missing)}")

    run_cli(args)


if __name__ == "__main__":
    main()
