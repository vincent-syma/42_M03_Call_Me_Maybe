"""State machine"""

from enum import Enum, auto
from typing import Any


class DecoderState(Enum):
    START = auto()
    AFTER_OPEN_BRACE = auto()
    AFTER_PROMPT_KEY = auto()
    AFTER_PROMPT_COLON = auto()
    AFTER_PROMPT = auto()
    AFTER_PROMPT_COMMA = auto()
    AFTER_NAME_KEY = auto()
    AFTER_NAME_COLON = auto()
    AFTER_FUNCTION_NAME = auto()
    AFTER_NAME_COMMA = auto()
    AFTER_PARAMS_KEY = auto()
    AFTER_PARAMS_COLON = auto()
    GENERATING_NUMBER = auto()
    GENERATING_INT = auto()
    GENERATING_STRING = auto()
    GENERATING_BOOLEAN = auto()
    AFTER_PARAMS_OPEN_BRACE = auto()
    AFTER_PARAM_KEY = auto()
    AFTER_PARAM_COLON = auto()
    AFTER_PARAM_VALUE = auto()
    AFTER_PARAM_COMMA = auto()
    CLOSE_PARAMS = auto()
    CLOSE_BRACE = auto()
    DONE = auto()


S = DecoderState


STATES: dict[DecoderState, dict[str, Any]] = {
    S.START: {
        "value": '{',
        "next_state": S.AFTER_OPEN_BRACE,
        "forced": True,
        "message": "Starting decoder",
    },
    S.AFTER_OPEN_BRACE: {
        "value": '"prompt"',
        "next_state": S.AFTER_PROMPT_KEY,
        "forced": True,
        "message": "Starting decoder",
    },
    S.AFTER_PROMPT_KEY: {
        "value": ':',
        "next_state": S.AFTER_PROMPT_COLON,
        "forced": True,
        "message": "Extracting prompt",
    },
    S.AFTER_PROMPT_COLON: {
        "value": '',                                           # empty string
        "next_state": S.AFTER_PROMPT,
        "forced": True,
        "message": "Extracting prompt",
    },
    S.AFTER_PROMPT: {
        "value": ',',
        "next_state": S.AFTER_PROMPT_COMMA,
        "forced": True,
        "message": "Extracting prompt",
    },
    S.AFTER_PROMPT_COMMA: {
        "value": '"name"',
        "next_state": S.AFTER_NAME_KEY,
        "forced": True,
        "message": "Selecting function",
    },
    S.AFTER_NAME_KEY: {
        "value": ':',
        "next_state": S.AFTER_NAME_COLON,
        "forced": True,
        "message": "Selecting function",
    },
    S.AFTER_NAME_COLON: {
        "value": '',                                           # empty string
        "next_state": S.AFTER_FUNCTION_NAME,
        "forced": True,
        "message": "Selecting function",
    },
    S.AFTER_FUNCTION_NAME: {
        "value": ',',
        "next_state": S.AFTER_NAME_COMMA,
        "forced": True,
        "message": "Selecting function",
    },
    S.AFTER_NAME_COMMA: {
        "value": '"parameters"',
        "next_state": S.AFTER_PARAMS_KEY,
        "forced": True,
        "message": "Extracting parameters",
    },
    S.AFTER_PARAMS_KEY: {
        "value": ':',
        "next_state": S.AFTER_PARAMS_COLON,
        "forced": True,
        "message": "Extracting parameters",
    },
    S.AFTER_PARAMS_COLON: {
        "value": '{',
        "next_state": S.AFTER_PARAMS_OPEN_BRACE,
        "forced": True,
        "message": "Extracting parameters",
    },
    S.GENERATING_BOOLEAN: {
        "value": '',
        "next_state": None,
        "forced": False,
        "message": "Generating boolean parameter",
    },
    S.GENERATING_NUMBER: {
        "value": '',
        "next_state": None,
        "forced": False,
        "message": "Generating number parameter",
    },
    S.GENERATING_INT: {
        "value": '',
        "next_state": None,
        "forced": False,
        "message": "Generating integer parameter",
    },
    S.GENERATING_STRING: {
        "value": '',
        "next_state": None,
        "forced": False,
        "message": "Generating string parameter",
    },
    S.AFTER_PARAMS_OPEN_BRACE: {
        "value": '',                                           # empty string
        "next_state": None,
        "forced": True,
        "message": "Extracting parameters",
    },
    S.AFTER_PARAM_KEY: {
        "value": ':',
        "next_state": S.AFTER_PARAM_COLON,
        "forced": True,
        "message": "Extracting parameters",
    },
    S.AFTER_PARAM_COLON: {
        "value": "",                                           # empty
        "next_state": None,
        "forced": True,
        "message": "Extracting parameters",
    },
    S.AFTER_PARAM_VALUE: {
        "value": "",                                           # empty
        "next_state": None,
        "forced": True,
        "message": "Extracting parameters",
    },
    S.AFTER_PARAM_COMMA: {
        "value": '',                                           # empty string
        "next_state": S.AFTER_PARAM_KEY,
        "forced": True,
        "message": "Extracting parameters",
    },
    S.CLOSE_PARAMS: {
        "value": '}',
        "next_state": S.CLOSE_BRACE,
        "forced": True,
        "message": "Done",
    },
    S.CLOSE_BRACE: {
        "value": '}',
        "next_state": S.DONE,
        "forced": True,
        "message": "Done",
    }
}
