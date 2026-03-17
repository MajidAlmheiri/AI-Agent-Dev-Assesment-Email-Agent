import os
import pandas as pd
from fastapi import FastAPI

app = FastAPI()

# This finds the 'data' folder regardless of where you run the script from
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, "data", "FreeZone_Companies_Data.csv")

@app.get("/companies")
def get_all_companies():
    df = pd.read_csv(CSV_PATH)
    return df.to_dict(orient="records")