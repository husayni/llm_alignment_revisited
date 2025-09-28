import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Union, overload

from openai import OpenAI
from openai.types.chat import ChatCompletion
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)


@dataclass
class Message:
    role: str
    content: str

    def to_dict(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class ConversationHistory:
    history: List[Message]


@dataclass
class ChatConfig:
    """Configuration for chat completions"""

    temperature: float = 0.0
    max_tokens: Optional[int] = 4096
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stream: bool = False
    response_format: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dict, excluding None values"""
        config = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "stream": self.stream,
        }
        if self.max_tokens is not None:
            config["max_tokens"] = self.max_tokens
        if self.response_format is not None:
            config["response_format"] = self.response_format
        return config


class LLMAgent:
    def __init__(
        self,
        model: str,
        system_prompt: Optional[str] = None,
        default_config: Optional[ChatConfig] = None,
        base_url: Optional[str] = "https://openrouter.ai/api/v1",
        name: Optional[str] = "LLM Agent",
    ):
        self.model = model
        self.default_config = default_config or ChatConfig()
        self.default_config = self.default_config.to_dict()

        assert os.environ.get("OPENROUTER_API_KEY"), (
            "Missing OpenRouter API key. Set OPENROUTER_API_KEY env var"
        )
        api_key = os.environ.get("OPENROUTER_API_KEY") if "openrouter" in base_url else "lmstudio"
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )
        self.system_prompt = system_prompt
        self.conversation_history: List[Message] = []
        logger.info(f"Initialized {name} with model: {self.model}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        retry=retry_if_exception_type(Exception),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def _make_api_call(
        self,
        messages: List[Dict[str, str]],
    ) -> str:
        try:
            response: ChatCompletion = self.client.chat.completions.create(
                model=self.model, messages=messages, **self.default_config
            )
            if response.choices and len(response.choices) > 0:
                content = response.choices[0].message.content
                return content if content else ""
            else:
                raise ValueError("No response content received")
        except Exception as e:
            logger.error(f"API call failed: {str(e)}")
            raise

    def instruct(self, instruction: str) -> str:
        messages = []

        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        messages.append({"role": "user", "content": instruction})
        response = self._make_api_call(messages)
        return response

    def chat(self, message: str, return_history: bool = False) -> Union[str, ConversationHistory]:
        if not self.conversation_history and self.system_prompt:
            self.conversation_history.append(Message("system", self.system_prompt))

        self.conversation_history.append(Message("user", message))

        messages = [msg.to_dict() for msg in self.conversation_history]

        response = self._make_api_call(messages)

        self.conversation_history.append(Message("assistant", response))

        if return_history:
            return response, ConversationHistory(
                history=self.conversation_history.copy(),
            )
        return response

    def get_history(self) -> List[Message]:
        return self.conversation_history.copy()

    def set_history(self, history: List[Message]):
        self.conversation_history = history.copy()

    def reset(self):
        self.conversation_history = []
