import os
import openai
from dotenv import load_dotenv

class OpenAIChat:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("OPENAI_API_KEY")
        openai.api_key = self.api_key

    def get_response(self, system_message, user_message):
        completion = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ]
        )
        return completion

if __name__ == "__main__":
    chat = OpenAIChat()
    system_message = "Your system message here"
    user_message = "Your user message here"
    response, prompt_tokens, completion_tokens = chat.get_response(system_message, user_message)
    print(response)
    print(prompt_tokens)
    print(completion_tokens)
