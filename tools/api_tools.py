from langchain_core.tools import tool
import pandas as pd
import requests

@tool
def count_companies(freezone: str, status: str, industry: str) -> int:
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
    
    # Filtering Free Zone company names and status
    temp_companies_filter1 = ((df["Free Zone"] == freezone) & (df["Status"] == status))
    
    # Filtering out non-logistics companies
    # I noticed that the logistics companies the word "logistics" in their and/or in the activity
    # I hope I understood this correctly.
    # Comment out the line below if you're looking for all active companies in JAFZA. Also, remove it from line 31
    temp_companies_filter2 = (df["Company Name"].str.contains(industry, case = False, na = False)) | (df["Activity"].str.contains(industry, case = False, na = False))
    
    # This contains the list of active logistics companies in JAFZA
    filtered_companies = df[temp_companies_filter1 & temp_companies_filter2]
    
    # Returns the number of active logistics companies in JAFZA 
    return len(filtered_companies)
    