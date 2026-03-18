import os
from pathlib import Path
from difflib import get_close_matches

import pandas as pd
import requests
from langchain_core.tools import tool


BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "data" / "FreeZone_Companies_Data.csv"
API_URL = os.getenv("COMPANIES_API_URL", "http://localhost:8000/companies")
DEFAULT_LICENSE_FEE_AED = float(os.getenv("LICENSE_FEE_AED", "15000"))


def _load_company_dataframe() -> pd.DataFrame:
    try:
        response = requests.get(API_URL, timeout=6)
        response.raise_for_status()
        raw_data = response.json()
        if isinstance(raw_data, list) and raw_data:
            return pd.DataFrame(raw_data)
    except Exception:
        pass

    return pd.read_csv(CSV_PATH)


def _normalize_text_series(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def _build_freezone_filter(df: pd.DataFrame, freezone: str) -> pd.Series:
    requested_freezone = str(freezone).strip()
    freezone_series = _normalize_text_series(df["Free Zone"])

    exact_match_filter_freezone = freezone_series.str.lower() == requested_freezone.lower()
    if exact_match_filter_freezone.any():
        return exact_match_filter_freezone

    known_freezones = sorted(set(freezone_series.tolist()))
    close_matches = get_close_matches(requested_freezone, known_freezones, n=1, cutoff=0.6)
    if close_matches:
        matched_freezone = close_matches[0]
        return freezone_series.str.lower() == matched_freezone.lower()

    return exact_match_filter_freezone


def _build_industry_filter(df: pd.DataFrame, industry: str) -> pd.Series | bool:
    if industry and str(industry).strip():
        search_term = str(industry).strip()
        return (
            df["Company Name"].fillna("").astype(str).str.contains(search_term, case=False, na=False)
        ) | (
            df["Activity"].fillna("").astype(str).str.contains(search_term, case=False, na=False)
        )
    return True

@tool
def count_companies(freezone: str, status: str = "Active", industry: str = "") -> int:
    """
    This method will be used to return the number of companies matching specific criteria.
    The method requires a freezone name, status, and industry.
    """

    df = _load_company_dataframe()

    freezone_filter = _build_freezone_filter(df, freezone)

    if status and str(status).strip():
        status_filter = _normalize_text_series(df["Status"]).str.lower() == str(status).strip().lower()
    else:
        status_filter = True

    industry_filter = _build_industry_filter(df, industry)

    filtered_companies = df[freezone_filter & status_filter & industry_filter]

    return len(filtered_companies)

@tool
def get_setup_estimates(freezone: str, industry: str, target_employees: int = 0) -> dict:
    """
    Calculates average capital and setup cost ranges for a specific freezone and industry.
    If the user specifies how many people they plan to hire, pass that number to 'target_employees'.
    """
    df = _load_company_dataframe()

    freezone_filter = _build_freezone_filter(df, freezone)
    industry_filter = _build_industry_filter(df, industry)

    filtered_df = df[freezone_filter & industry_filter].copy()

    if filtered_df.empty:
        return {"error": f"No matching data found for the industry keyword: '{industry}'"}

    filtered_df["Capital (AED)"] = pd.to_numeric(filtered_df["Capital (AED)"].astype(str).str.replace(",", ""), errors="coerce").fillna(0)
    filtered_df["Employees"] = pd.to_numeric(filtered_df["Employees"].astype(str).str.replace(",", ""), errors="coerce").fillna(0)

    avg_capital = filtered_df["Capital (AED)"].mean()

    if target_employees > 0:
        calc_employees = target_employees
        employee_note = f"Calculated based on your requested {target_employees} employees."
    else:
        calc_employees = filtered_df["Employees"].mean()
        employee_note = f"Calculated based on the industry average of {round(calc_employees)} employees."

    avg_license_fee = DEFAULT_LICENSE_FEE_AED
    visa_costs = calc_employees * 4500

    total_base_cost = avg_capital + avg_license_fee + visa_costs
    min_range = total_base_cost * 0.9
    max_range = total_base_cost * 1.1

    matched_freezones = df.loc[freezone_filter, "Free Zone"].dropna().astype(str).str.strip()
    matched_freezone = matched_freezones.iloc[0] if not matched_freezones.empty else str(freezone).strip()

    return {
        "matched_freezone": matched_freezone,
        "matched_industry": str(industry).strip(),
        "average_license_fee_estimate": f"{avg_license_fee:,.2f} AED",
        "average_capital": f"{avg_capital:,.2f} AED",
        "employees_used_for_math": round(calc_employees),
        "employee_note": employee_note,
        "license_fee_estimate": f"{avg_license_fee:,.2f} AED",
        "visa_budget_estimate": f"{visa_costs:,.2f} AED",
        "total_estimated_setup": f"{total_base_cost:,.2f} AED",
        "setup_cost_range": f"{min_range:,.0f} - {max_range:,.0f} AED"
    }