import os

import requests
from dotenv import load_dotenv

# Load FARMI_API_URL / FARMI_API_KEY from the project-root .env.
# Never hardcode secrets in source files (this repo is public).
load_dotenv()

# FARMI configuration
base_url = os.getenv("FARMI_API_URL", "https://gptlab.rd.tuni.fi/students/ollama/v1").rstrip("/")
url = f"{base_url}/completions"
api_key = os.getenv("FARMI_API_KEY", "")
if not api_key:
    raise SystemExit("FARMI_API_KEY is not set. Add it to a .env file at the project root (see README).")

# Headers
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
}

# Your prompt
prompt = input("Enter your prompt: ")

# Request data
data = {
    "model": "llama3.3:70b",
    "prompt": prompt,
    "max_tokens": 200,
    "temperature": 0.7,
}

# Send request
print("\nSending request to FARMI...")
response = requests.post(url, headers=headers, json=data)

# Display results
print(f"\nStatus Code: {response.status_code}")
if response.status_code == 200:
    result = response.json()
    print("\nResponse:")
    print(result["choices"][0]["text"])
    print(f"\nTokens used: {result['usage']['total_tokens']}")
else:
    print(f"Error: {response.text}")
