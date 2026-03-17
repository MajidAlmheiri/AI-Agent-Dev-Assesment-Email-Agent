from langchain_core.tools import tool
import pandas as pd
import requests
from difflib import get_close_matches

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

    exact_match_filter = freezone_series.str.lower() == requested_freezone.lower()
    if exact_match_filter.any():
        freezone_filter = exact_match_filter
    else:
        known_freezones = sorted(set(freezone_series.tolist()))
        close_matches = get_close_matches(requested_freezone, known_freezones, n=1, cutoff=0.6)
        if close_matches:
            matched_freezone = close_matches[0]
            freezone_filter = freezone_series.str.lower() == matched_freezone.lower()
        else:
            freezone_filter = exact_match_filter

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
    