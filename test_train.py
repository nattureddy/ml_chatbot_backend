import requests

# Upload the CSV file
files = {'file': open('test.csv', 'rb')}

# Set the correct target column and model
data = {
    'target_column': 'pci',  # Column you're predicting
    'model_name': 'Linear Regression'  # Or: Decision Tree, Random Forest, KNN
}

# Make the request to the /train endpoint
response = requests.post("http://127.0.0.1:8000/train", files=files, data=data)

# Print response
print("Status Code:", response.status_code)
print("Response:", response.json())
