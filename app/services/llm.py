from abc import ABC, abstractmethod
from groq import Groq
from app.config import settings

class LLMProvider(ABC):
    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        pass

class GroqProvider(LLMProvider):
    def __init__(self):
        self.client = Groq(api_key=settings.GROQ_API_KEY)
        self.model = "llama-3.3-70b-specdec" 

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