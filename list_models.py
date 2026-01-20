import os
import sys
import json
import requests

api_key = "AIzaSyC_DvpSty1ScEfUmDfkFJZuJ1PUzKy5XQo"
if not api_key:
    print("ERROR: GOOGLE_API_KEY environment variable is not set.")
    sys.exit(1)

endpoints = [
    "https://generativelanguage.googleapis.com/v1/models",
    "https://generativelanguage.googleapis.com/v1beta/models",
    "https://generativelanguage.googleapis.com/v1beta/openai/models",
    "https://generativelanguage.googleapis.com/v1/openai/models",
    "https://generativelanguage.googleapis.com/v1/models/list",
]

headers = {"Authorization": f"Bearer {api_key}"}

for url in endpoints:
    print('\n---')
    print(f"Trying URL: {url}")
    try:
        r = requests.get(url, headers=headers, timeout=15)
    except Exception as e:
        print(f"Request failed: {e}")
        continue
    print(f"Status: {r.status_code}")
    # Try to print JSON prettily if possible, but avoid printing huge content
    try:
        body = r.json()
        print(json.dumps(body, indent=2)[:10000])
    except Exception:
        print(r.text[:10000])

# Also try using the OpenAI-compatible path under the openai prefix
openai_like = "https://generativelanguage.googleapis.com/v1beta/openai/models"
print('\n---')
print(f"Trying OpenAI-compatible URL: {openai_like} with api key as query param")
try:
    r2 = requests.get(openai_like, params={"key": api_key}, timeout=15)
    print(f"Status: {r2.status_code}")
    try:
        print(json.dumps(r2.json(), indent=2)[:10000])
    except Exception:
        print(r2.text[:10000])
except Exception as e:
    print(f"Request failed: {e}")
