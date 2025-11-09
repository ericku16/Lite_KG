import os
import json
from abc import ABC, abstractmethod
from dotenv import load_dotenv

load_dotenv()

from openai import OpenAI
import ollama

class BaseLLMClient(ABC):
    """
    定義一個統一的 'chat' 介面
    """
    def __init__(self, model_name: str):
        self.model = model_name

    @abstractmethod
    def chat(self, system_prompt: str, user_content: str, is_json: bool = False) -> str:
        """
        Args:
            system_prompt: 系統提示
            user_content: 用戶輸入
            is_json: 是否強制要求 JSON 輸出

        Returns:
            模型回傳的內容 (str)
        """
        pass

class OpenAIClient(BaseLLMClient):
    """
    OpenAI API 客戶端。
    """
    def __init__(self, model_name: str, api_key: str):
        super().__init__(model_name)
        if not api_key:
            raise ValueError("OpenAI API key is required.")
        self.client = OpenAI(api_key=api_key)

    def chat(self, system_prompt: str, user_content: str, is_json: bool = False) -> str:
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ]
            
            response_format = {"type": "json_object"} if is_json else {"type": "text"}
            
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format=response_format
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"❌ OpenAI API call failed: {e}")
            return ""

class OllamaClient(BaseLLMClient):
    """
    Ollama (Mistral) 客戶端。
    """
    def __init__(self, model_name: str):
        super().__init__(model_name)
        try:
            ollama.list()
        except Exception as e:
            print(f"⚠️ Warning: Could not connect to Ollama. Is it running? {e}")

    def chat(self, system_prompt: str, user_content: str, is_json: bool = False) -> str:
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ]
            
            format_type = "json" if is_json else ""
            
            response = ollama.chat(
                model=self.model,
                messages=messages,
                format=format_type
            )
            return response['message']['content']
        except Exception as e:
            print(f"❌ Ollama API call failed: {e}")
            return ""

def get_llm_client(provider: str, model_name: str, api_key: str = None) -> BaseLLMClient:
    """
    根據 'provider' 決定執行哪個 LLM 客戶端
    """
    if provider == "openai":
        return OpenAIClient(model_name=model_name, api_key=api_key)
    elif provider == "ollama":
        return OllamaClient(model_name=model_name)
    else:
        raise ValueError(f"Unknown provider: {provider}. Supported providers are 'openai', 'ollama'.")

if __name__ == "__main__":
    print("--- Testing OpenAI ---")
    openai_client = get_llm_client(
        provider="openai", 
        model_name="gpt-4o", 
        api_key=os.getenv("OPENAI_API_KEY")
    )
    if openai_client:
        print(openai_client.chat("You are a helpful assistant.", "Say 'hello'."))

    print("\n--- Testing Ollama ---")
    ollama_client = get_llm_client(
        provider="ollama", 
        model_name="mistral:latest"
    )
    print(ollama_client.chat("You are a helpful assistant.", "Say 'hello'."))