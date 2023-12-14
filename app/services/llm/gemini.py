# Or use `os.getenv('GOOGLE_API_KEY')` to fetch an environment variable.
from common.config import settings
import google.generativeai as genai

GOOGLE_API_KEY = settings.GEMINI_API_KEY

genai.configure(api_key=GOOGLE_API_KEY)
