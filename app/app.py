# Importing required modules
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import shutil
import imghdr
import process_data
import build_model

# Run with uvicorn app:app

# Initialize the FastAPI app
app = FastAPI()

# Ensure the FastAPI server allows requests from the frontend origin by configuring CORS properly
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.retrain_countdown = 10

# Define the endpoint for GET requests
@app.get("/create_data")
def create_data_endpoint():
    # Call the function from create_data.py to create data
    data = create_data.create()
    return {"message": "Data created successfully", "data": data}

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    if imghdr.what(file.file) is None:
        raise HTTPException(status_code=400, detail="Le fichier téléchargé n'est pas une image valide.")
    with open(f"data/{file.filename}", "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    # Call the function from process_data.py to process data
    result = process_data.predict_and_save()
    app.retrain_countdown -= 1
    if app.retrain_countdown == 0:
        background_tasks.add_task(build_model.build_model)
        app.retrain_countdown = 10
    print("retrain_countdown: ", app.retrain_countdown)
    return {"message": "Data processed successfully", "result": result}