"""Pydantic data models for function calling input and output."""

from typing import Protocol, cast, Any
from enum import Enum
from pydantic import BaseModel, ConfigDict, Field, field_validator


class LLMBackend(Protocol):
    def get_path_to_vocab_file(self) -> str:
        ...

    def encode(self, text: str) -> Any:
        ...

    def decode(self, ids: Any) -> str:
        ...

    def get_logits_from_input_ids(
        self,
        input_ids: list[int],
        past_key_values: Any,
    ) -> tuple[list[float], object]:
        ...


class LLMModel(BaseModel):
    """Validated wrapper around a callable LLM backend implementation."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    backend: Any = Field(
        ...,
        description="Underlying LLM implementation"
    )

    @field_validator("backend")
    @classmethod
    def validate_backend(cls, value: Any) -> Any:
        """Ensure the wrapped object exposes the methods the decoder needs."""
        # if isinstance(value, LLMModel):
        #     value = value.backend

        required_methods = (
            "get_path_to_vocab_file",
            "encode",
            "decode",
            "get_logits_from_input_ids",
        )
        missing_methods = [
            name for name in required_methods
            if not callable(getattr(value, name, None))
        ]
        if missing_methods:
            raise ValueError(
                "model must provide callable methods "
                f"{required_methods}; missing {missing_methods}"
            )
        return value

    def get_path_to_vocab_file(self) -> str:
        backend = cast(LLMBackend, self.backend)
        return backend.get_path_to_vocab_file()

    def encode(self, text: str) -> list[int]:
        backend = cast(LLMBackend, self.backend)
        result = backend.encode(text).tolist()

        if not isinstance(result, list) or not result:
            raise ValueError("LLM method encode() returned invalid output")

        if not isinstance(result[0], list):
            raise ValueError(
                "LLM method encode() expected batched output"
            )

        return result[0]

    def decode(self, ids: list[int]) -> str:
        if not isinstance(ids, list) or not ids:
            # debug:
            print(ids)
            raise ValueError("decode() expected non-empty list[int]")
        if not all(isinstance(i, int) for i in ids):
            raise ValueError("decode() expected list[int]")
        backend = cast(LLMBackend, self.backend)
        result = backend.decode(ids)
        if not isinstance(result, str):
            raise ValueError(
                "LLM method decode() returned invalid output"
            )
        return result

    def get_logits_from_input_ids(
            self,
            input_ids: list[int],
            past_key_values: Any = None
    ) -> tuple[list[float], object]:
        backend = cast(LLMBackend, self.backend)
        return backend.get_logits_from_input_ids(input_ids, past_key_values)


class ParameterType(str, Enum):
    """Supported parameter types for function definitions."""
    number = "number"
    string = "string"
    boolean = "boolean"
    integer = "integer"


class FunctionParameter(BaseModel):
    """A single function parameter with its type."""
    type: ParameterType


class TestPrompt(BaseModel):
    """A natural language prompt to be translated into a function call."""
    prompt: str


class FunctionDefinition(BaseModel):
    """Represents a callable function with typed parameters."""
    name: str
    description: str
    parameters: dict[str, FunctionParameter]
    returns: FunctionParameter

    # @field_validator('name')
    # @classmethod
    # def name_must_start_with_fn(cls, nm: str) -> str:
    #     """Validate that function name starts with fn_."""
    #     if not nm.startswith('fn_'):
    #         raise ValueError(
    #             f"Function name must start with 'fn_', got: {nm}"
    #             )
    #     return nm


class Result(BaseModel):
    """Represents the output of a function call."""
    prompt: str
    name: str
    parameters: dict[str, str | float | int | bool]
