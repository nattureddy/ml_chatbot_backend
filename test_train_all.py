import requests

files = {'file': open('test.csv', 'rb')}
data = {
    'target_column': ' pci'  # include the leading space exactly as it is
}

response = requests.post("http://127.0.0.1:8000/train_all", files=files, data=data)

print("Status Code:", response.status_code)
print("Response:", response.json())
