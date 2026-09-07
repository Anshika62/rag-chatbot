import base64
import os

import requests
from dotenv import load_dotenv

load_dotenv()

account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID")
api_token = os.getenv("CLOUDFLARE_API_TOKEN")

url = (
    f"https://api.cloudflare.com/client/v4/accounts/"
    f"{account_id}/ai/run/@cf/black-forest-labs/flux-1-schnell"
)

headers = {
    "Authorization": f"Bearer {api_token}",
    "Content-Type": "application/json",
}

payload = {
    "prompt": "A realistic futuristic city at sunset, cinematic lighting, highly detailed",
    "steps": 4,
}

response = requests.post(
    url,
    headers=headers,
    json=payload,
    timeout=120,
)

print("HTTP status:", response.status_code)

if not response.ok:
    print(response.text)
    raise SystemExit(1)

result = response.json()

if not result.get("success"):
    print(result)
    raise SystemExit(1)

image_base64 = result["result"]["image"]

image_bytes = base64.b64decode(image_base64)

output_file = "test_flux_image.jpg"

with open(output_file, "wb") as file:
    file.write(image_bytes)

print(f"Image generated successfully: {output_file}")
print(f"Image size: {len(image_bytes)} bytes")