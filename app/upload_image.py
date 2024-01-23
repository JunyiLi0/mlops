import requests


url = "http://localhost:8000/upload/"
file_path = "/mnt/c/Users/ASUS/OneDrive/Bureau/Cours/MLOPS/test 0.png"

with open(file_path, "rb") as file:
    response = requests.post(url, files={"file": file})

print(response.json())