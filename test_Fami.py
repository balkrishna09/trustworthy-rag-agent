import requests
import json

# FARMI configuration
url = "https://gptlab.rd.tuni.fi/students/ollama/v1/completions"
api_key = "sk-ollama-gptlab-toufique-student-63aca62c5c1497228ab5a487037cea15"

# Headers
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

# Your prompt
prompt = input("Enter your prompt: ")

# Request data
data = {
    "model": "llama3.3:70b",
    "prompt": prompt,
    "max_tokens": 200,
    "temperature": 0.7
}

# Send request
print("\nSending request to FARMI...")
response = requests.post(url, headers=headers, json=data)

# Display results
print(f"\nStatus Code: {response.status_code}")
if response.status_code == 200:
    result = response.json()
    print(f"\nResponse:")
    print(result['choices'][0]['text'])
    print(f"\nTokens used: {result['usage']['total_tokens']}")
else:
    print(f"Error: {response.text}")