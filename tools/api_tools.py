from langchain_core.tools import tool
import pandas as pd
import requests
from difflib import get_close_matches

# TO-DO: A helper function could help to make filtering logic shorter since a lot of lines are basically identical...

@tool
def count_companies(freezone: str, status: str = "Active", industry: str = "") -> int:
    """
    This method will be used to return the number of companies matching specific criteria.
    The method requires a freezone name, status, and industry.
    """

    # Getting data from our api
    url = "http://localhost:8000/companies"
    response = requests.get(url)
    
    # Unpacking the data and turning it into a python dictionary
    raw_data = response.json()
    
    # Creating data frame
    df = pd.DataFrame(raw_data)
    
    # Filtering Free Zone company names and status (default status=Active)
    requested_freezone = str(freezone).strip()
    freezone_series = df["Free Zone"].fillna("").astype(str).str.strip()

    exact_match_filter_freezone = freezone_series.str.lower() == requested_freezone.lower()
    if exact_match_filter_freezone.any():
        freezone_filter = exact_match_filter_freezone
    else:
        known_freezones = sorted(set(freezone_series.tolist()))
        close_matches = get_close_matches(requested_freezone, known_freezones, n=1, cutoff=0.6)
        if close_matches:
            matched_freezone = close_matches[0]
            freezone_filter = freezone_series.str.lower() == matched_freezone.lower()
        else:
            freezone_filter = exact_match_filter_freezone

    if status and str(status).strip():
        status_filter = df["Status"].str.strip().str.lower() == str(status).strip().lower()
    else:
        status_filter = True
    
    # Filtering out non-logistics companies
    # I noticed that the logistics companies the word "logistics" in their and/or in the activity
    # I hope I understood this correctly.
    # Comment out the line below if you're looking for all active companies in JAFZA. Also, remove it from line 31
    if industry and str(industry).strip():
        industry_filter = (
            df["Company Name"].str.contains(industry, case=False, na=False)
        ) | (
            df["Activity"].str.contains(industry, case=False, na=False)
        )
    else:
        industry_filter = True
    
    # This contains the list of active logistics companies in JAFZA
    filtered_companies = df[freezone_filter & status_filter & industry_filter]
    
    # Returns the number of active logistics companies in JAFZA 
    return len(filtered_companies)

@tool
def get_setup_estimates(freezone: str, industry: str, target_employees: int = 0) -> dict:
    """
    Calculates average capital and setup cost ranges for a specific freezone and industry.
    If the user specifies how many people they plan to hire, pass that number to 'target_employees'.
    """
    # Getting data from our api
    url = "http://localhost:8000/companies"
    response = requests.get(url)
    raw_data = response.json()
    df = pd.DataFrame(raw_data)

    # Filtering Free Zone
    requested_freezone = str(freezone).strip()
    freezone_series = df["Free Zone"].fillna("").astype(str).str.strip()
    exact_match_filter_freezone = freezone_series.str.lower() == requested_freezone.lower()
    if exact_match_filter_freezone.any():
        freezone_filter = exact_match_filter_freezone
    else:
        known_freezones = sorted(set(freezone_series.tolist()))
        close_matches = get_close_matches(requested_freezone, known_freezones, n=1, cutoff=0.6)
        if close_matches:
            matched_freezone = close_matches[0]
            freezone_filter = freezone_series.str.lower() == matched_freezone.lower()
        else:
            freezone_filter = exact_match_filter_freezone
     
    # Filtering industry (Keyword Search)       
    if industry and str(industry).strip():
        search_term = str(industry).strip()
        industry_filter = (
            df["Company Name"].str.contains(search_term, case=False, na=False)
        ) | (
            df["Activity"].str.contains(search_term, case=False, na=False)
        )
    else:
        industry_filter = True
                
    # Perform Calculations
    filtered_df = df[freezone_filter & industry_filter].copy()
    
    if filtered_df.empty:
        return {"error": f"No matching data found for the industry keyword: '{industry}'"}

    filtered_df["Capital (AED)"] = pd.to_numeric(filtered_df["Capital (AED)"], errors='coerce').fillna(0)
    filtered_df["Employees"] = pd.to_numeric(filtered_df["Employees"], errors='coerce').fillna(0)

    avg_capital = filtered_df["Capital (AED)"].mean()
    
    # --- NEW LOGIC: Use user's employee count if provided, otherwise use average ---
    if target_employees > 0:
        calc_employees = target_employees
        employee_note = f"Calculated based on your requested {target_employees} employees."
    else:
        calc_employees = filtered_df["Employees"].mean()
        employee_note = f"Calculated based on the industry average of {round(calc_employees)} employees."
    
    # Business Logic Estimates
    license_fee = 15000 
    visa_costs = calc_employees * 4500 # 4500 AED per visa
    
    total_base_cost = avg_capital + license_fee + visa_costs
    min_range = total_base_cost * 0.9
    max_range = total_base_cost * 1.1

    return {
        "matched_freezone": str(df[freezone_filter]["Free Zone"].iloc[0]),
        "matched_industry": industry,
        "average_capital": f"{avg_capital:,.2f} AED",
        "employees_used_for_math": round(calc_employees),
        "employee_note": employee_note,
        "license_fee_estimate": f"{license_fee:,.2f} AED",
        "visa_budget_estimate": f"{visa_costs:,.2f} AED",
        "total_estimated_setup": f"{total_base_cost:,.2f} AED",
        "setup_cost_range": f"{min_range:,.0f} - {max_range:,.0f} AED"
    }