from groq import Groq

from app.config import settings
from app.models.schemas import LLMProvider


class GroqProvider(LLMProvider):
    def __init__(self):
        self.client = Groq(api_key=settings.GROQ_API_KEY)
        self.model = settings.LLM_MODEL

    @property
    def provider_name(self) -> str:
        return "groq"

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
        )
        return completion.choices[0].message.content


def get_llm():
    if settings.LLM_PROVIDER == "groq":
        return GroqProvider()
    raise ValueError(f"Unsupported provider: {settings.LLM_PROVIDER}")
