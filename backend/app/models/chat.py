from enum import StrEnum
from time import time
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt


class ChatRole(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class ChatMessage(BaseModel):
    """The common message shape sent to and received from chat providers."""

    model_config = ConfigDict(extra="allow")

    role: ChatRole
    content: str | list[dict[str, Any]] | None
    name: str | None = None


class ChatCompletionRequest(BaseModel):
    """Supported non-streaming subset of OpenAI's chat completion request."""

    model_config = ConfigDict(extra="allow")

    model: str = Field(min_length=1)
    messages: list[ChatMessage] = Field(min_length=1)
    temperature: float | None = Field(default=None, ge=0, le=2)
    max_tokens: NonNegativeInt | None = None
    stream: Literal[False] = False

    def to_provider_payload(self, provider_model_name: str) -> dict[str, Any]:
        """Keep caller-supplied message fields while replacing NOMINAL's model alias."""
        payload = self.model_dump(exclude_none=True)
        payload["model"] = provider_model_name
        return payload


class TokenUsage(BaseModel):
    input_tokens: NonNegativeInt = 0
    output_tokens: NonNegativeInt = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class InternalChatChoice(BaseModel):
    index: NonNegativeInt = 0
    message: ChatMessage
    finish_reason: str | None = None


class InternalChatCompletion(BaseModel):
    """Provider-neutral response used between adapters and the gateway."""

    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid4().hex}")
    created: int = Field(default_factory=lambda: int(time()))
    model: str
    choices: list[InternalChatChoice] = Field(min_length=1)
    usage: TokenUsage = Field(default_factory=TokenUsage)
    provider_latency_ms: NonNegativeInt


class OpenAIChatCompletionResponse(BaseModel):
    """Wire model returned to clients of `/v1/chat/completions`."""

    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[InternalChatChoice]
    usage: dict[str, int]

    @classmethod
    def from_internal(cls, completion: InternalChatCompletion) -> "OpenAIChatCompletionResponse":
        return cls(
            id=completion.id,
            created=completion.created,
            model=completion.model,
            choices=completion.choices,
            usage={
                "prompt_tokens": completion.usage.input_tokens,
                "completion_tokens": completion.usage.output_tokens,
                "total_tokens": completion.usage.total_tokens,
            },
        )
