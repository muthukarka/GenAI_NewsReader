import os
from dotenv import load_dotenv

load_dotenv()  # Load .env variables

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("ERROR: API Key not found. Make sure the .env file is correctly loaded.")
else:
    print(f"API Key loaded successfully: {api_key[:5]}********")  # Partial key for security
