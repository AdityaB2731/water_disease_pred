import requests

try:
    r = requests.get('https://api.groq.ai/v1/parse')
    print(r.status_code, r.text)
except Exception as e:
    print(e)
