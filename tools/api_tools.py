import os
from pathlib import Path
from difflib import get_close_matches

import pandas as pd
import requests
from langchain_core.tools import tool

# ----------------------------
# Environment & File Paths
# ----------------------------
# Set up the paths to our data and API. 
# We use os.getenv to pull from the .env file, but provide safe defaults if they are missing.
BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "data" / "FreeZone_Companies_Data.csv"
API_URL = os.getenv("COMPANIES_API_URL", "http://localhost:8000/companies")
DEFAULT_LICENSE_FEE_AED = float(os.getenv("LICENSE_FEE_AED", "15000"))


# ----------------------------
# Core Helper: Data Loader
# ----------------------------
def _load_company_dataframe() -> pd.DataFrame:
    """
    Attempts to fetch live data from the FastAPI server.
    If the server is down or times out, it safely falls back to reading the local CSV.
    """
    try:
        response = requests.get(API_URL, timeout=6)
        response.raise_for_status()
        raw_data = response.json()
        if isinstance(raw_data, list) and raw_data:
            return pd.DataFrame(raw_data)
    except Exception:
        # If anything goes wrong with the API, silently pass and use the CSV fallback below.
        pass

    return pd.read_csv(CSV_PATH)


# ----------------------------
# Helper: Data Cleaner
# ----------------------------
def _normalize_text_series(series: pd.Series) -> pd.Series:
    """
    Cleans up a column of text data by replacing empty cells (NaN) with empty strings,
    and stripping out accidental spaces from the beginning or end of words.
    """
    return series.fillna("").astype(str).str.strip()


# ----------------------------
# Helper: Free Zone Filter (With Fuzzy Matching)
# ----------------------------
def _build_freezone_filter(df: pd.DataFrame, freezone: str) -> pd.Series:
    """
    Creates a True/False mask to filter the dataset down to a specific free zone.
    It first tries an exact match. If the user makes a typo, it uses fuzzy matching 
    (difflib) to find the closest known free zone in the database.
    """
    requested_freezone = str(freezone).strip()
    freezone_series = _normalize_text_series(df["Free Zone"])

    # Attempt 1: Exact lowercase match
    exact_match_filter_freezone = freezone_series.str.lower() == requested_freezone.lower()
    if exact_match_filter_freezone.any():
        return exact_match_filter_freezone

    # Attempt 2: Fuzzy matching for typos (e.g., "Jabal Ali" -> "Jebel Ali")
    known_freezones = sorted(set(freezone_series.tolist()))
    close_matches = get_close_matches(requested_freezone, known_freezones, n=1, cutoff=0.6)
    if close_matches:
        matched_freezone = close_matches[0]
        return freezone_series.str.lower() == matched_freezone.lower()

    # Fallback: Return the empty exact match if nothing is close
    return exact_match_filter_freezone


# ----------------------------
# Helper: Industry Search
# ----------------------------
def _build_industry_filter(df: pd.DataFrame, industry: str) -> pd.Series | bool:
    """
    Searches for an industry keyword in BOTH the 'Company Name' and 'Activity' columns.
    Returns True/False for every row depending on if the keyword was found.
    """
    if industry and str(industry).strip():
        search_term = str(industry).strip()
        # The | symbol acts as an "OR". Keep the row if the keyword is in the Name OR the Activity.
        return (
            df["Company Name"].fillna("").astype(str).str.contains(search_term, case=False, na=False)
        ) | (
            df["Activity"].fillna("").astype(str).str.contains(search_term, case=False, na=False)
        )
    
    # If no industry was provided, don't filter anything out.
    return True


# ----------------------------
# TOOL 1: The Counter (From Agent 1)
# ----------------------------
# I thought keeping this tool is a good idea since it'll make our LLM more versatile
@tool
def count_companies(freezone: str, status: str = "Active", industry: str = "") -> int:
    """
    This method will be used to return the number of companies matching specific criteria.
    The method requires a freezone name, status, and industry.
    """
    # 1. Load the data
    df = _load_company_dataframe()

    # 2. Build the three filters
    freezone_filter = _build_freezone_filter(df, freezone)

    if status and str(status).strip():
        status_filter = _normalize_text_series(df["Status"]).str.lower() == str(status).strip().lower()
    else:
        status_filter = True

    industry_filter = _build_industry_filter(df, industry)

    # 3. Apply the filters to shrink the dataset
    filtered_companies = df[freezone_filter & status_filter & industry_filter]

    # 4. Return the final integer count to the LLM
    return len(filtered_companies)


# ----------------------------
# TOOL 2: The Calculator (Agent 2)
# ----------------------------
@tool
def get_setup_estimates(freezone: str, industry: str, target_employees: int = 0) -> dict:
    """
    Calculates average capital and setup cost ranges for a specific freezone and industry.
    If the user specifies how many people they plan to hire, pass that number to 'target_employees'.
    """
    # 1. Load and filter the data
    df = _load_company_dataframe()
    freezone_filter = _build_freezone_filter(df, freezone)
    industry_filter = _build_industry_filter(df, industry)
    
    filtered_df = df[freezone_filter & industry_filter].copy()

    # If the search turns up empty, tell the LLM so it can inform the user
    if filtered_df.empty:
        return {"error": f"No matching data found for the industry keyword: '{industry}'"}

    # 2. Clean up numbers (Remove commas from money strings and convert to actual floats)
    filtered_df["Capital (AED)"] = pd.to_numeric(filtered_df["Capital (AED)"].astype(str).str.replace(",", ""), errors="coerce").fillna(0)
    filtered_df["Employees"] = pd.to_numeric(filtered_df["Employees"].astype(str).str.replace(",", ""), errors="coerce").fillna(0)

    # 3. Calculate the averages from the dataset
    avg_capital = filtered_df["Capital (AED)"].mean()

    # 4. Handle Employee Math (User request vs. Industry average)
    if target_employees > 0:
        calc_employees = target_employees
        employee_note = f"Calculated based on your requested {target_employees} employees."
    else:
        calc_employees = filtered_df["Employees"].mean()
        employee_note = f"Calculated based on the industry average of {round(calc_employees)} employees."

    # 5. Calculate Final Costs
    avg_license_fee = DEFAULT_LICENSE_FEE_AED
    visa_costs = calc_employees * 4500 # Assume 4500 AED per visa

    total_base_cost = avg_capital + avg_license_fee + visa_costs
    
    # Create a realistic ±10% buffer for the estimate range
    min_range = total_base_cost * 0.9
    max_range = total_base_cost * 1.1

    # 6. Extract the correctly formatted Free Zone name for the final output
    matched_freezones = df.loc[freezone_filter, "Free Zone"].dropna().astype(str).str.strip()
    matched_freezone = matched_freezones.iloc[0] if not matched_freezones.empty else str(freezone).strip()

    # 7. Return the structured data dictionary to the LLM to write the email
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
