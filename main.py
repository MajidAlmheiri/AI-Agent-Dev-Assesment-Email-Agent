import pandas as pd
from pathlib import Path

# Main function
def main():
    # Getting the path of the dataset
    # This implementation should work regardless of the OS used
    file_path = Path("data") / "FreeZone_Companies_Data.csv"

    # Declaring dataframe to store the dataset csv file
    df = pd.read_csv(file_path)
    
    count_freezones_by_status(df, "Jebel Ali Free Zone", "Active")

# Function to be called by Agent 1
def count_freezones_by_status(df, freezone: str, status: str, display_info: bool=False) -> int: #Maybe remove the data type constrains..?
    """
    This method returns the number of freezones with the specified status
    This method will be used by Agent 1
    
    :param df: dataframe (contains the contents of the csv file.). Always has to be included
    :param freezone: freezone name
    :type freezone: str
    :param status: freezone status (active, inactive, pending_renewal)
    :type status: str
    :param display_info: set to true to display all the info for the companies
    :type status: str
    """
    
    if (display_info == False):
        return len(df[(df["Free Zone"] == freezone) & (df["Status"] == status)])
    
    else:
        print(df[(df["Free Zone"] == freezone) & (df["Status"] == status)])
        return len(df[(df["Free Zone"] == freezone) & (df["Status"] == status)])


if __name__ == "__main__":
    main()