import os
from ollama import Client


class OllamaService:

    def __init__(self):

        api_key = os.getenv("OLLAMA_API_KEY")

        if not api_key:
            raise ValueError(
                "OLLAMA_API_KEY is not configured."
            )

        self.client = Client(
            host="https://ollama.com",
            headers={
                "Authorization": f"Bearer {api_key}"
            }
        )

        self.model = os.getenv(
            "OLLAMA_MODEL",
            "gpt-oss:20b-cloud"
        )

    def generate(self, prompt):

        response = self.client.chat(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return response["message"]["content"]