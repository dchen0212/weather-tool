import re
from dataclasses import dataclass, field
from typing import Any

import pandas as pd


DEFAULT_COMPARE_FIELD_CANDIDATES = [
    "t_avg",
    "t_max",
    "t_min",
    "precip",
    "solar_rad",
]


@dataclass
class AgentPlan:
    intent: str
    lat: float | None = None
    lon: float | None = None
    start_date: str | None = None
    end_date: str | None = None
    unit: str = "C"
    compare_field: str | None = None
    missing: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    next_actions: list[str] = field(default_factory=list)
    raw_prompt: str = ""


def _find_float_after_keywords(text: str, keywords: list[str]) -> float | None:
    pattern = rf"(?:{'|'.join(re.escape(k) for k in keywords)})\s*[:=]?\s*(-?\d+(?:\.\d+)?)"
    match = re.search(pattern, text, flags=re.IGNORECASE)
    return float(match.group(1)) if match else None


def _find_dates(text: str) -> list[str]:
    return re.findall(r"\b\d{4}-\d{2}-\d{2}\b", text)


def _detect_unit(text: str) -> str:
    lowered = text.lower()
    if "kelvin" in lowered or re.search(r"\bunit\s*[:=]?\s*k\b", lowered):
        return "K"
    return "C"


def _detect_compare_field(text: str) -> str | None:
    lowered = text.lower()
    for field in DEFAULT_COMPARE_FIELD_CANDIDATES:
        if field in lowered:
            return field
    return None


def _build_next_actions(intent: str) -> list[str]:
    if intent == "compare_predictions":
        return [
            "Upload the real and predicted CSV files.",
            "Pick a shared numeric field for evaluation.",
            "Review MAE, RMSE, R², and interval-level errors.",
        ]
    if intent == "summarize_weather":
        return [
            "Fetch the requested NASA POWER weather data.",
            "Generate a compact summary of major variables.",
            "Review the summary and export the CSV if needed.",
        ]
    return [
        "Fetch the requested NASA POWER weather data.",
        "Review the returned table and summary.",
        "Export CSV or continue with comparison analysis.",
    ]


def build_agent_plan(prompt: str) -> AgentPlan:
    cleaned = prompt.strip()
    lowered = cleaned.lower()

    compare_keywords = ["compare", "comparison", "预测", "对比", "误差", "mae", "rmse", "r2"]
    summarize_keywords = ["summarize", "summary", "insight", "report", "概括", "总结", "分析天气趋势"]
    fetch_keywords = ["fetch", "get", "download", "weather", "天气", "nasa", "power"]

    if any(keyword in lowered for keyword in compare_keywords):
        intent = "compare_predictions"
    elif any(keyword in lowered for keyword in summarize_keywords):
        intent = "summarize_weather"
    elif any(keyword in lowered for keyword in fetch_keywords):
        intent = "fetch_weather"
    else:
        intent = "fetch_weather"

    dates = _find_dates(cleaned)
    lat = _find_float_after_keywords(cleaned, ["lat", "latitude", "纬度"])
    lon = _find_float_after_keywords(cleaned, ["lon", "longitude", "经度"])

    plan = AgentPlan(
        intent=intent,
        lat=lat,
        lon=lon,
        start_date=dates[0] if len(dates) >= 1 else None,
        end_date=dates[1] if len(dates) >= 2 else None,
        unit=_detect_unit(cleaned),
        compare_field=_detect_compare_field(cleaned),
        next_actions=_build_next_actions(intent),
        raw_prompt=cleaned,
    )

    if intent in {"fetch_weather", "summarize_weather"}:
        if plan.lat is None:
            plan.missing.append("latitude")
        if plan.lon is None:
            plan.missing.append("longitude")
        if plan.start_date is None:
            plan.missing.append("start_date")
        if plan.end_date is None:
            plan.missing.append("end_date")
        if not plan.missing:
            plan.notes.append("Ready to fetch NASA POWER weather data.")
            if intent == "summarize_weather":
                plan.notes.append("A summary report will be generated after retrieval.")
    elif intent == "compare_predictions":
        if plan.compare_field is None:
            plan.notes.append("No explicit comparison field found; the UI can fall back to a shared numeric column.")
        plan.notes.append("Needs uploaded real/predicted CSV files in the Streamlit app.")

    return plan


def summarize_plan(plan: AgentPlan) -> dict[str, Any]:
    return {
        "intent": plan.intent,
        "lat": plan.lat,
        "lon": plan.lon,
        "start_date": plan.start_date,
        "end_date": plan.end_date,
        "unit": plan.unit,
        "compare_field": plan.compare_field,
        "missing": plan.missing,
        "notes": plan.notes,
        "next_actions": plan.next_actions,
    }


def summarize_weather_frame(df: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {"rows": len(df), "columns": list(df.columns)}

    if "date" in df.columns and not df.empty:
        summary["date_range"] = {
            "start": str(df["date"].iloc[0]),
            "end": str(df["date"].iloc[-1]),
        }

    numeric_candidates = [
        "t_avg",
        "t_max",
        "t_min",
        "precip",
        "solar_rad",
        "rel_humidity",
        "wind_speed_10m",
    ]

    variable_summary: dict[str, Any] = {}
    for column in numeric_candidates:
        if column in df.columns:
            values = pd.to_numeric(df[column], errors="coerce").dropna()
            if values.empty:
                continue
            variable_summary[column] = {
                "mean": round(float(values.mean()), 3),
                "min": round(float(values.min()), 3),
                "max": round(float(values.max()), 3),
            }
    summary["variables"] = variable_summary
    return summary


def build_summary_highlights(summary: dict[str, Any]) -> list[str]:
    highlights: list[str] = []
    date_range = summary.get("date_range")
    if date_range:
        highlights.append(f"Covers {summary.get('rows', 0)} daily records from {date_range['start']} to {date_range['end']}.")

    variables = summary.get("variables", {})
    if "t_avg" in variables:
        highlights.append(
            f"Average daily temperature is {variables['t_avg']['mean']} in the selected unit."
        )
    if "precip" in variables:
        highlights.append(
            f"Daily precipitation ranges from {variables['precip']['min']} to {variables['precip']['max']}."
        )
    if "solar_rad" in variables:
        highlights.append(
            f"Solar radiation has a mean of {variables['solar_rad']['mean']}."
        )

    return highlights
