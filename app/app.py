# Importing required modules
from fastapi import FastAPI
import create_data
import process_data

# Initialize the FastAPI app
app = FastAPI()

# Define the endpoint for GET requests
@app.get("/create_data")
def create_data_endpoint():
    # Call the function from create_data.py to create data
    data = create_data.create()
    return {"message": "Data created successfully", "data": data}

# Define the endpoint for POST requests
@app.post("/process_data")
def process_data_endpoint():
    # Call the function from process_data.py to process data
    result = process_data.process()
    return {"message": "Data processed successfully", "result": result}
