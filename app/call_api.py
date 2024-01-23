import requests

# Make a GET request
response = requests.get('http://localhost:8000/create_data')
print(response.json())

# Make a POST request
response = requests.post('http://localhost:8000/process_data')
print(response.json())
