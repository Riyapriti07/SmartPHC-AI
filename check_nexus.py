"""
Free connection check for the Nexus API (run this BEFORE starting the app).

  python check_nexus.py

Step 1 lists your available models (no tokens used).
Step 2 sends one tiny message (about 15-20 tokens) to prove the key works.
It tries the base URL from .env first, then the same URL with/without "/v1".
"""
import os, sys
import requests
from dotenv import load_dotenv

load_dotenv()
key = (os.environ.get("NEXUS_API_KEY") or "").strip()
base = (os.environ.get("NEXUS_BASE_URL") or "https://nexusapi.navigatelabs.ai/v1").strip().rstrip("/")
model = (os.environ.get("NEXUS_MODEL") or "gemini-2.5-flash").strip()

if not key or key == "your_api_key_here":
    sys.exit("X  NEXUS_API_KEY is empty. Paste your key into the .env file first.")

candidates = [base]
alt = base[:-3] if base.endswith("/v1") else base + "/v1"
candidates.append(alt)
H = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

working = None
for b in candidates:
    print(f"\nTrying base URL: {b}")
    try:
        r = requests.get(b + "/models", headers=H, timeout=20)
    except requests.RequestException as e:
        print("   cannot connect:", type(e).__name__, "(check internet / try a mobile hotspot)")
        continue
    print("   GET /models ->", r.status_code)
    if r.status_code == 200:
        working = b
        try:
            ids = [m.get("id") for m in r.json().get("data", [])]
            print(f"   {len(ids)} models available. First ones:", ", ".join(ids[:15]))
            print("   Your NEXUS_MODEL is:", model, "->", "AVAILABLE" if model in ids else "NOT in the list! Pick one from above.")
        except Exception:
            pass
        break
    if r.status_code in (401, 403):
        sys.exit("X  The server rejected your API key (401/403). Re-copy it from Model Hub > Key Management.")

if not working:
    sys.exit("\nX  Could not reach a working endpoint. Send the lines above to your guide/me.")

print("\nSending one tiny test message...")
r = requests.post(working + "/chat/completions", headers=H, timeout=45,
                  json={"model": model, "max_tokens": 10, "temperature": 0,
                        "messages": [{"role": "user", "content": "Say OK"}]})
print("   POST /chat/completions ->", r.status_code)
if r.status_code == 200:
    d = r.json()
    print("   Reply:", d["choices"][0]["message"]["content"].strip())
    print("   Tokens used:", d.get("usage"))
    print("\nOK  SUCCESS. Put this in .env:  NEXUS_BASE_URL=" + working)
else:
    print("   Server said:", r.text[:300])
    sys.exit("\nX  Chat failed. If it says model not found, change NEXUS_MODEL to one from the list above.")
