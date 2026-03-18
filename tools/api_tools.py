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
def get_setup_estimates(freezone: str, industry: str) -> dict:
    """
    Calculates average capital and setup cost ranges for a specific freezone and industry.
    Use this when a user asks about the costs or capital requirements for opening a company.
    """
    # Getting data from our api
    url = "http://localhost:8000/companies"
    response = requests.get(url)
    
    # Unpacking the data and turning it into a python dictionary
    raw_data = response.json()
    
    # Creating data frame
    df = pd.DataFrame(raw_data)

    # Filtering Free Zone company names and industry
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
     
    # Filtering industry (Activity)       
    requested_industry = str(industry).strip()
    industry_series = df["Activity"].fillna("").astype(str).str.strip()

    exact_match_filter_industry = industry_series.str.lower() == requested_industry.lower()
    if exact_match_filter_industry.any():
        industry_filter = exact_match_filter_industry
    else:
        known_industries = sorted(set(industry_series.tolist()))
        close_matches = get_close_matches(requested_industry, known_industries, n=1, cutoff=0.6)
        if close_matches:
            matched_industry = close_matches[0]
            industry_filter = industry_series.str.lower() == matched_industry.lower()
        else:
            industry_filter = exact_match_filter_industry
                
            
    # Perform Calculations
    filtered_df = df[freezone_filter & industry_filter].copy()
    
    if filtered_df.empty:
        return {"error": "No matching data found for this specific inquiry."}

    # Ensure numbers are treated as numbers
    filtered_df["Capital (AED)"] = pd.to_numeric(filtered_df["Capital (AED)"], errors='coerce').fillna(0)
    filtered_df["Employees"] = pd.to_numeric(filtered_df["Employees"], errors='coerce').fillna(0)

    avg_capital = filtered_df["Capital (AED)"].mean()
    avg_employees = filtered_df["Employees"].mean()
    
    # Business Logic Estimates
    license_fee = 15000 # Flat estimate for the assessment
    visa_costs = avg_employees * 4500 # Approx cost per residency visa
    
    # Calculating the total base cost and the range for the agent
    # I used a 10% margin for the range calculation
    # The requirment didn't specify a specific margin, so I hope this is fine
    total_base_cost = avg_capital + license_fee + visa_costs
    min_range = total_base_cost * 0.9
    max_range = total_base_cost * 1.1

    return {
        "matched_freezone": str(df[freezone_filter]["Free Zone"].iloc[0]),
        "matched_industry": industry,
        "average_capital": f"{avg_capital:,.2f} AED",
        "average_employees": round(avg_employees),
        "license_fee_estimate": f"{license_fee:,.2f} AED",
        "visa_budget_estimate": f"{visa_costs:,.2f} AED",
        "total_estimated_setup": f"{total_base_cost:,.2f} AED",
        "setup_cost_range": f"{min_range:,.0f} - {max_range:,.0f} AED"
    }