import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
key = os.getenv("GROQ_API_KEY")

if not key or key.strip() in ["", "api_key"]:
    raise RuntimeError("GROQ_API_KEY missing or still placeholder. Update .env with your real key.")

client = Groq(api_key=key)

resp = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[{"role":"user","content":"Reply with exactly 5 words: Groq API is working."}],
    temperature=0
)

print(resp.choices[0].message.content)
