from fastapi import FastAPI
import pandas as pd
from pathlib import Path

# Create the web app
app = FastAPI()

# FastAPI will automatically turn the returned list into JSON
@app.get("/companies")
def get_all_companies():
    
    # Getting the path of the dataset
    # This implementation should work regardless of the OS used
    file_path = Path("data") / "FreeZone_Companies_Data.csv"

    # Declaring dataframe to store the dataset csv file
    df = pd.read_csv(file_path)
    
    # Convert the Pandas DataFrame into standard Python dictionaries
    # orient="records" tells pandas to make a list of rows
    data_dict = df.to_dict(orient="records")
    
    # Returns the data back
    return data_dict
    