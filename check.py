import requests, os, json
from dotenv import load_dotenv
load_dotenv()
api_key = os.environ.get("GROQ_API_KEY")
response = requests.get(
    "https://api.groq.com/openai/v1/models",
    headers={"Authorization": f"Bearer {api_key}"}
)
print(response.status_code)
print(json.dumps(response.json(), indent=2))
