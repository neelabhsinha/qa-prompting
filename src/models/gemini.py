import os
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold


class Gemini:
    def __init__(self, model_name='gemini-1.5-flash'):
        self.model_name = model_name
        api_key = os.getenv('GOOGLE_API_KEY')
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name=model_name)
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

    def __call__(self, text):
        response = self.model.generate_content([text], safety_settings=self.safety_settings)
        return response.text
