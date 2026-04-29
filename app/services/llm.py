from app.config import settings
from app.models.schemas import LLMProvider


class GroqProvider(LLMProvider):
    def __init__(self):
        try:
            from groq import Groq
        except ImportError as exc:  # pragma: no cover - depends on local installation
            raise RuntimeError(
                "The groq package is not installed. Install it before using the Groq provider."
            ) from exc
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


class MockProvider(LLMProvider):
    @property
    def provider_name(self) -> str:
        return "mock"

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        return "Mock provider response. Replace LLM_PROVIDER with 'groq' for live generation."


def get_llm():
    if settings.LLM_PROVIDER == "groq":
        return GroqProvider()
    if settings.LLM_PROVIDER == "mock":
        return MockProvider()
    raise ValueError(f"Unsupported provider: {settings.LLM_PROVIDER}")
