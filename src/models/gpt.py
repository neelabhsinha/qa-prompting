from openai import OpenAI
import os


class GPT:
    def __init__(self, model='gpt-3.5-turbo'):
        api_key = os.getenv('OPENAI_API_KEY')
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.instruction = 'You are an AI assistant designed to help users in completion of a task.'

    def __call__(self, text):
        messages = [
            {'role': 'system', 'content': self.instruction},
            {'role': 'user', 'content': text}
        ]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages
        )
        output = response.choices[0].message.content
        return output
