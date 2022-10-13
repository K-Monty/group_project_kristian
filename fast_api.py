import requests
import pandas as pd

url = "http://127.0.0.1:8000/n_chunks"
response = requests.get(url)
print(response.json()["chunks"])

url = "http://127.0.0.1:8000/chunk/3"
response = requests.get(url)
json_text = response.text

short = json_text[1:-1]
short = short.replace("\\", "")

df = pd.read_json(short, orient="records")
print(df.head())
print(df.shape)