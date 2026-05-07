import re
from dataclasses import dataclass, field
from typing import Any


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


def build_agent_plan(prompt: str) -> AgentPlan:
    cleaned = prompt.strip()
    lowered = cleaned.lower()

    compare_keywords = ["compare", "comparison", "预测", "对比", "误差", "mae", "rmse", "r2"]
    fetch_keywords = ["fetch", "get", "download", "weather", "天气", "nasa", "power"]

    if any(keyword in lowered for keyword in compare_keywords):
        intent = "compare_predictions"
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
        raw_prompt=cleaned,
    )

    if intent == "fetch_weather":
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
    }
