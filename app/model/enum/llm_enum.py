
from enum import Enum

class LLMProviders(Enum):
    GROQ = "groq"
    OPENAI = "openai"
    GEMINI = "gemini"

    def get_available_models(self):
        if self == LLMProviders.GROQ:
            return ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
        elif self == LLMProviders.OPENAI:
            return ["gpt-4o", "gpt-4.1-mini"]
        elif self == LLMProviders.GEMINI:
            return ["gemini-2.0-pro", "gemini-1.5-pro", "gemini-2.5-flash"]
        else:
            raise ValueError(f"Unsupported LLM: {self.value}")

    def get_llm_instance(self, model_name: str, temperature: float = 0.2):
        if self == LLMProviders.GROQ:
            from langchain_groq import ChatGroq
            return ChatGroq(model=model_name, temperature=temperature)
        elif self == LLMProviders.OPENAI:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(model=model_name, temperature=temperature)
        elif self == LLMProviders.GEMINI:
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(model=model_name, temperature=temperature)
        else:
            raise ValueError(f"Unsupported LLM: {self.value}")