import requests
import json

# Load the best model name saved from /train_all
with open("last_model.json", "r") as f:
    best_model_name = json.load(f)["model_name"]

# Prepare file and model name for prediction
files = {'file': open('predict_input.csv', 'rb')}
data = {'model_name': best_model_name}

# Call the /predict endpoint
response = requests.post("http://127.0.0.1:8000/predict", files=files, data=data)

# Show the response
print("Status Code:", response.status_code)
print("Response:", response.json())
