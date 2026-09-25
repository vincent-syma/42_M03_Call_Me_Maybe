"""Decoder"""

# --- IMPORTS ---

import json              # JSON output
import numpy as np       # logit stats
from typing import Any

from llm_sdk import Small_LLM_Model as LLM  # default LLM

# type validation:
from pydantic import (
    BaseModel,
    ConfigDict,
    PrivateAttr,
    field_validator,
    Field
)

from .models import FunctionDefinition, ParameterType, LLMModel, Result
from .state_machine import STATES, DecoderState as S
from rich import print
from .format_rich import ITALIC, YELLOW, RESET  # print formatting
# from .status import update_status

# import time
import logging

logger = logging.getLogger(__name__)


# --- DECODER ---

class Decoder(BaseModel):
    """
    Constrained decoder for generating valid JSON function calls.

    Usage:
    # 1 - Decoder uses its default LLM (Qwen/Qwen3-0.6B)
    decoder = Decoder()

    # 2 - Decoder uses LLM model passed as argument
      (Required methods:
        - get_path_to_vocab_file(self) -> str,
        - encode(self, text: str) -> torch.Tensor | list[int],
        - decode(self, ids: torch.Tensor | list[int]) -> str,
        - get_logits_from_input_ids(self, input_ids: list[int]) -> list[float],
      )
    decoder = Decoder(model=model)

    output = decoder.decode(function_defs, test_prompt)

    FunctionDefinition (dict object) = decoder.select_function(
                                        function_defs,
                                        test_prompt
                                       )
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: LLMModel = Field(default_factory=lambda: LLMModel(
        # backend=LLM()
        backend=LLM(device="cpu")
    ))
    # device="cpu": to avoid CUDA mismatch of torch and device

    _state: S = PrivateAttr(default=S.START)
    _token_to_id: dict[str, int] = PrivateAttr(default_factory=dict)
    _id_to_token: dict[int, str] = PrivateAttr(default_factory=dict)
    _original_prompt: str | None = PrivateAttr(default=None)
    _selected_function: FunctionDefinition | None = PrivateAttr(default=None)
    _current_param: str | None = PrivateAttr(default=None)
    _current_param_index: int = PrivateAttr(default=0)
    _input_ids: list[int] = PrivateAttr(default_factory=list)
    _current_value_tokens: list[int] = PrivateAttr(default_factory=list)
    _collected_params: dict[str, str | float | bool] = PrivateAttr(
        default_factory=dict
    )
    _last_status = PrivateAttr(default=None)
    _valid_tokens: dict[str, list[int]] = PrivateAttr(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        """
        Initialize private attributes that do not need Pydantic validation
        """

        self._token_to_id, self._id_to_token = self._load_vocabulary(
            self.model.get_path_to_vocab_file()
        )
        self._forced_tokens: list[int] = []
        self._forced_position: int = 0
        self._generated_tokens: list[int] = []
        self._past_key_values: Any = None
        self._warnings: list[str] = []

        self._valid_tokens = {
            'number': [
                tid for tid, token in self._id_to_token.items()
                if token.strip().replace('.', '', 1).isdigit()
            ],
            'plus_minus': [
                tid for tid, token in self._id_to_token.items()
                if token.strip()
                and all(c in '+-' for c in token.strip())
            ],
            'string_start': [
                tid for tid, token in self._id_to_token.items()
                if token
                and not token.startswith(" ")
                and token not in {'"', ' '}
                # {'{', '}', '[', ']'}  #, ':', ','}
            ],
            'string_continue': [
                tid for tid, token in self._id_to_token.items()
                if token.strip()
                and token != '"'
            ],
            "dot": (
                [tid]
                if (tid := self._get_token_id_or_none(".")) is not None
                else []
            ),
            "comma": (
                [tid]
                if (tid := self._get_token_id_or_none(",")) is not None
                else []
            ),
            "close_brace": (
                [tid]
                if (tid := self._get_token_id_or_none("}")) is not None
                else []
            ),
            "quote": (
                [tid]
                if (tid := self._get_token_id_or_none('"')) is not None
                else []
            ),
            "true": (
                [tid]
                if (tid := self._get_token_id_or_none("true")) is not None
                else []
            ),
            "false": (
                [tid]
                if (tid := self._get_token_id_or_none("false")) is not None
                else []
            ),
        }

    @field_validator("model", mode="before")
    @classmethod
    def wrap_model(cls, value: Any) -> LLMModel:
        if isinstance(value, LLMModel):
            return value
        return LLMModel(backend=value)

    def decode(
            self,
            function_defs: list[FunctionDefinition],
            original_prompt: str,
    ) -> Any:
        """Generate a function call result for the given prompt."""
        # RESET STATE with new calling - but it should be clear anyway
        self._reset_state()
        self._original_prompt = original_prompt

        try:
            self._input_ids = self.model.encode(
                self._build_prompt(
                    function_defs,
                    fn_selection=False
                )
            )

            MAX_TOKENS = 248
            initial_length = len(self._input_ids)

            # # TIME DEBUG:
            # start = time.time()

            while self._state != S.DONE:
                if len(self._input_ids) > initial_length + MAX_TOKENS:
                    raise ValueError("Max token limit reached")

                # DEBUG: print current state
                logger.debug("State: %s", self._state.name)

                self._navigate_state(function_defs)

                # # TIME DEBUG ---------------

                # elapsed = time.time() - start

                # logger.debug(
                #     "state=%s,\ttime=%.4fs",
                #     self._state, elapsed,
                # )

                # # --------------------------

            self._validate_result(function_defs)

            # just for the terminal text right now:
            output = self._build_result()

        finally:
            # RESET after everything is done again
            self._reset_state()

        try:
            return json.loads(output)
        except json.JSONDecodeError as e:
            raise e

    def _get_output(self) -> str:
        return self.model.decode(self._generated_tokens)

    def _reset_state(self) -> None:
        """Reset private attributes to default values"""
        self._state = S.START
        self._selected_function = None
        self._current_param = None
        self._current_param_index = 0
        self._input_ids = []
        self._current_value_tokens = []
        self._collected_params = {}
        self._last_status = None
        self._forced_tokens = []
        self._forced_position = 0
        self._generated_tokens = []
        self._warnings = []
        # cache:
        self._past_key_values = None

    def _build_prompt(
            self,
            functions: list[FunctionDefinition],
            prompt: str | None = None,
            fn_selection: bool = False
    ) -> str:
        """
        Build a prompt for the LLM to select and call a function.
        Prompt structure:
        - What the user wants
        - Available functions (names, descriptions, param names and types)
        - What format to output

        Arguments:
        - functions: list of available functions to choose from
        - prompt: user request
        - fn_selection: if the prompt is used for function selection
        """

        if self._original_prompt is None:
            raise ValueError("No prompt to process")
        elif prompt is None:
            prompt = self._original_prompt

        functions_text = "\n".join(
            self._format_function(fn) for fn in functions
        )

        fn_selection_prompt = (
            f"Prompt: {prompt}\n\n"
            f"Available functions:\n{functions_text}\n\n"
            "Choose exactly one function that best matches the 'prompt'.\n"
        )

        param_selection_addition = (
            "Extract the parameter values from the 'prompt'.\n\n"
        )

        if fn_selection:
            return fn_selection_prompt

        return fn_selection_prompt + param_selection_addition

    @staticmethod
    def _format_function(fn: FunctionDefinition) -> str:
        """Format a single function definition as readable text."""
        params = ", ".join(
            f"{param_name}: {param.type.value}"
            for param_name, param in fn.parameters.items()
        )
        return f"- {fn.name}({params}): {fn.description}"

    def _navigate_state(
            self,
            functions: list[FunctionDefinition],
    ) -> None:
        """
        Generate next token through LLM and append.
        If the value is complete, advance state.
        """

        # special case - model state beggining with forced token:
        if self._state == S.AFTER_PARAM_COLON:
            if self._selected_function is None:
                raise ValueError("No function selected")
            param_name = list(
                self._selected_function.parameters.keys()
            )[self._current_param_index]
            self._current_param = param_name
            param_type = self._selected_function.parameters[param_name].type

            if param_type == ParameterType.string:
                # forced opening quote
                next_token = self._model_generation(functions)
                self._input_ids.append(next_token)
                self._generated_tokens.append(next_token)

                # DEBUG:
                logger.debug(
                    "- APPENDED(X): %r",
                    self._id_to_token[next_token]
                )

                # debug: for some reason it does not happen elsewhere:
                self._forced_position += 1

                if self._forced_position >= len(self._forced_tokens):
                    self._forced_tokens = []
                    self._forced_position = 0

            if self._handle_branching_state_transition() is True:
                return

        state = STATES[self._state]

        # states not producing any value, just directing to another state
        if state["value"] is None:
            if self._handle_branching_state_transition() is True:
                return

        # model generated next token according to the state valid tokens
        next_token = self._model_generation(functions)

        if not state["forced"]:
            self._current_value_tokens.append(next_token)

        current_tokens = self._current_value_tokens.copy()

        # LLM generating parameter values:
        # - still generating or finished?

        if self._state == S.GENERATING_BOOLEAN:
            self._input_ids.append(next_token)
            self._generated_tokens.append(next_token)

            # DEBUG:
            logger.debug(
                    "- APPENDED(B): %r",
                    self._id_to_token[next_token]
                )

            # counting on correct tokenizing, maybe i should trim this too??
            if (current_tokens == self._valid_tokens['true']
                    or current_tokens == self._valid_tokens['false']):
                self._finalize_current_value()
                self._set_new_state(S.AFTER_PARAM_VALUE)

                return

        elif self._state == S.GENERATING_STRING:
            token_text = self._id_to_token[next_token]

            if '"' in token_text:
                quote_pos = token_text.index('"')
                last_text_part = self.model.encode(token_text[:quote_pos])
                self._generated_tokens.extend(last_text_part)
                self._input_ids.extend(last_text_part)

                if len(current_tokens) > 1:
                    value_str = self.model.decode(current_tokens[:-1])
                else:
                    value_str = ""
                value_str += token_text[:quote_pos]

                logger.debug("BEFORE: %r", value_str)
                value_str = value_str.replace("Ġ", " ").strip()
                logger.debug("AFTER: %r", value_str)
                # value_str = value_str.strip()

                # self._finalize_current_value(original_prompt)

                if self._current_param is None:
                    raise ValueError(
                        "Current_param is None during value collection"
                        )
                if not self._selected_function:
                    raise ValueError("No function selected.")
                param = (
                    self._selected_function.parameters[self._current_param]
                )
                param_type = param.type
                # value = self.model.encode(value_str)
                if not self._validate_generated_value(
                    self._current_param, param_type, value_str,
                ):
                    raise ValueError(
                        "Unable to fill required parameter "
                        f"'{self._current_param}' for function "
                        f"'{self._selected_function.name}' from the prompt"
                    )
                self._collected_params[self._current_param] = value_str
                self._current_value_tokens = []

                self._set_new_state(S.AFTER_PARAM_VALUE)
                return

            else:
                self._input_ids.append(next_token)
                self._generated_tokens.append(next_token)

            # DEBUG:
            logger.debug(
                    "- APPENDED(S): %r",
                    self._id_to_token[next_token]
                )

        elif (self._state == S.GENERATING_NUMBER
              or self._state == S.GENERATING_INT):
            token_text = self._id_to_token[next_token]
            # termination tokens for number:
            if token_text in {",", "}"}:

                # remove the termination token from current_value_tokens
                self._current_value_tokens.pop()

                if "," in token_text:
                    pos = token_text.index(',')

                if "}" in token_text:
                    pos = token_text.index('}')

                # reencode the trimmed token to id and add
                last_text_part = self.model.encode(token_text[:pos])
                self._generated_tokens.extend(last_text_part)
                self._input_ids.extend(last_text_part)

                self._finalize_current_value()
                self._set_new_state(S.AFTER_PARAM_VALUE)

                return

            else:
                self._input_ids.append(next_token)
                self._generated_tokens.append(next_token)

            # DEBUG:
            logger.debug(
                    "- APPENDED(N): %r",
                    self._id_to_token[next_token]
                )

        # FORCED STATES:
        else:
            self._input_ids.append(next_token)
            self._generated_tokens.append(next_token)

            # DEBUG:
            logger.debug(
                    "- APPENDED(E): %r",
                    self._id_to_token[next_token]
                )

            self._forced_position += 1

            if self._forced_position >= len(self._forced_tokens):
                self._forced_tokens = []
                self._forced_position = 0

                if state["value"] == "":
                    if self._handle_branching_state_transition() is True:
                        return

                if (STATES[self._state] == state
                        and state["next_state"] is not None):
                    self._set_new_state(state["next_state"])
                    return

    def _handle_branching_state_transition(self) -> bool:
        """Handle branching state transitions.
        Returns True if changed the state, False if not."""
        if self._state == S.AFTER_PARAMS_OPEN_BRACE:
            if self._selected_function is None:
                raise ValueError("No selected function")

            if not self._selected_function.parameters:
                self._set_new_state(S.CLOSE_BRACE)
            else:
                self._set_new_state(S.AFTER_PARAM_KEY)
            return True

        elif self._state == S.AFTER_PARAM_COLON:
            if self._selected_function is None:
                raise ValueError("No function selected")
            param_name = list(
                self._selected_function.parameters.keys()
            )[self._current_param_index]
            self._current_param = param_name
            param_type = self._selected_function.parameters[param_name].type

            if param_type == ParameterType.number:
                self._set_new_state(S.GENERATING_NUMBER)
            elif param_type == ParameterType.integer:
                self._set_new_state(S.GENERATING_INT)
            elif param_type == ParameterType.string:
                self._set_new_state(S.GENERATING_STRING)
            elif param_type == ParameterType.boolean:
                self._set_new_state(S.GENERATING_BOOLEAN)
            return True

        elif self._state == S.AFTER_PARAM_VALUE:
            self._current_param_index += 1

            if self._selected_function is None:
                raise ValueError("No selected function")

            if (self._current_param_index
                    < len(self._selected_function.parameters)):
                self._set_new_state(S.AFTER_PARAM_COMMA)
            else:
                self._set_new_state(S.CLOSE_BRACE)
            return True
        return False

    def _set_new_state(self, new_state: S) -> None:
        """
        Set the decoder state and update the status message if it has changed.
        """
        # if new_state is None:
        #     self._state = STATES[self._state]["next_state"]
        # else:
        self._state = new_state

        # if self._state != S.DONE:
        #     state =  STATES[self._state]
        #     message = state["message"]
        #     if message != self._last_status:
        #         # # overwriting the status message in the console
        #         # update_status(message)
        #         self._last_status = message

    def select_function(
            self,
            functions: list[FunctionDefinition],
    ) -> FunctionDefinition:
        """
        Select the most likely function for the prompt
        using log-probability scoring.
        """

        prefix_ids = self.model.encode(
            self._build_prompt(functions, fn_selection=True)
        )

        scored_functions = []

        for fn in functions:
            candidate_ids = self.model.encode(fn.name.strip())
            log_prob = self._sequence_logprob(prefix_ids, candidate_ids)
            scored_functions.append((log_prob, fn))
            # print(f"  Function '{fn.name}'\tlog-prob: {log_prob:.4f}")

        _, selected_function = max(scored_functions, key=lambda item: item[0])

        return selected_function

    def _sequence_logprob(
            self,
            prefix_ids: list[int],
            candidate_ids: list[int]
    ) -> float:
        """
        Compute the log-probability of a candidate sequence given a prefix.
        """
        current_ids = list(prefix_ids)
        log_prob = 0.0

        for token_id in candidate_ids:
            # logits = np.array(
            #     self.model.get_logits_from_input_ids(current_ids)
            # )
            l, _ = self.model.get_logits_from_input_ids(current_ids)
            logits = np.array(l)

            logits = logits - np.max(logits)
            probs = np.exp(logits)
            probs = probs / probs.sum()

            p = float(probs[int(token_id)])
            p = max(p, 1e-20)

            log_prob += np.log(p)
            current_ids.append(int(token_id))

        return log_prob

    def _model_generation(
            self,
            functions: list[FunctionDefinition],
    ) -> int:
        """
        Ask model for next token,
        mask invalid ones,
        pick best,
        advance state.
        """

        # return probabilities of tokens in vocabulary
        # logits = np.array(
        #     self.model.get_logits_from_input_ids(self._input_ids)
        # )

        # cached:
        if self._past_key_values is None:
            input_ids = self._input_ids
        else:
            input_ids = [self._input_ids[-1]]

        l, self._past_key_values = (
            self.model.get_logits_from_input_ids(
                input_ids,
                self._past_key_values,
            )
        )
        logits = np.array(l)

        valid_ids = self._get_valid_tokens(functions)
        if not valid_ids:
            # return
            raise ValueError(
                f"No valid tokens for state {self._state}"
            )

        # create mask for each token
        mask = np.full(len(logits), True)

        # turn the mask off for valid tokens
        mask[valid_ids] = False

        # set invalid tokens to -infinity, so they are not chosen
        logits[mask] = -np.inf

        # pick the token with highest probability (valid)
        next_token = int(np.argmax(logits))

        # # DEBUG:
        # logger.debug(
        #             "CHOSEN: %r",
        #             self._id_to_token[next_token]
        #         )

        return next_token

    def _load_vocabulary(
            self,
            path: str
    ) -> tuple[dict[str, int], dict[int, str]]:
        """
        Load vocabulary file and return token_to_id and id_to_token dicts.
        """

        with open(path) as file:
            token_to_id = json.load(file)
        id_to_token = {value: key for key, value in token_to_id.items()}
        return token_to_id, id_to_token

    def _get_token_ids_for_string(self, text: str) -> list[int]:
        """Return the list of token IDs that represent this string."""
        try:
            result: list[int] = self.model.encode(text)
            return result
        except (AttributeError, KeyError, IndexError, TypeError):
            return []

    # ???
    def _get_token_id_or_none(self, token: str) -> int | None:
        """Return the token ID for a given token, or None if not found."""
        token_ids = self._get_token_ids_for_string(token)
        return token_ids[0] if token_ids else None

    def _get_valid_tokens(
            self,
            functions: list[FunctionDefinition],
    ) -> list[int]:
        """Return valid token IDs for the current decoder state."""
        pos = len(self._current_value_tokens)
        state = STATES[self._state]
        if state["value"] is None:
            return []
        string = state["value"]

        # LLM deciding states:

        if self._state == S.GENERATING_BOOLEAN:
            valid = []
            if pos < len(self._valid_tokens['true']):
                valid.append(self._valid_tokens['true'][pos])
            if pos < len(self._valid_tokens['false']):
                valid.append(self._valid_tokens['false'][pos])
            return valid

        elif self._state == S.GENERATING_NUMBER:
            if not self._current_value_tokens:
                return (
                    self._valid_tokens['number']
                    + self._valid_tokens['dot']
                )

            current_value = self.model.decode(self._current_value_tokens)

            if "." not in current_value:
                return (
                    self._valid_tokens['number']
                    + self._valid_tokens['dot']
                )

            if current_value.endswith("."):
                return self._valid_tokens['number']

            return (
                self._valid_tokens['number']
                + self._valid_tokens['comma']
                + self._valid_tokens['close_brace']
            )

        elif self._state == S.GENERATING_INT:
            if not self._current_value_tokens:
                return self._valid_tokens['number']

            return (
                self._valid_tokens['number']
                + self._valid_tokens['comma']
                + self._valid_tokens['close_brace']
            )

        elif self._state == S.GENERATING_STRING:
            if pos == 0:
                return self._valid_tokens['string_start']

            return (
                self._valid_tokens['string_continue']
                + self._valid_tokens['quote']
            )

        # FORCED STATES - preselected values:

        elif self._state == S.AFTER_PROMPT_COLON:
            string = json.dumps(self._original_prompt)
            # string = f'"{self._original_prompt}"'

        elif self._state == S.AFTER_NAME_COLON:
            if self._selected_function is None:
                self._selected_function = self.select_function(
                    functions)
            string = f'"{self._selected_function.name}"'

        elif self._state == S.AFTER_PARAMS_OPEN_BRACE:
            if self._selected_function is None:
                raise ValueError("No function selected")
            if not self._selected_function.parameters:
                string = '}'
            else:
                first_param = list(
                    self._selected_function.parameters.keys()
                )[0]
                string = f'"{first_param}"'

        elif self._state == S.AFTER_PARAM_COLON:
            if self._selected_function is None:
                raise ValueError("No function selected")
            if self._current_param is None:
                raise ValueError("No current parameter")

            param_name = list(
                self._selected_function.parameters.keys()
            )[self._current_param_index]
            self._current_param = param_name
            param_type = self._selected_function.parameters[param_name].type

            if param_type == ParameterType.string:
                # force opening quote
                string = '"'

        elif self._state == S.AFTER_PARAM_VALUE:
            if self._selected_function is None:
                raise ValueError("No function selected")
            if self._current_param is None:
                raise ValueError("No current parameter")
            param_name = list(
                self._selected_function.parameters.keys()
            )[self._current_param_index]
            self._current_param = param_name
            param_type = self._selected_function.parameters[param_name].type

            if param_type == ParameterType.string:
                string += '"'
            if (self._current_param_index
                    < len(self._selected_function.parameters) - 1):
                string += ','
            else:
                string += '}'

        elif self._state == S.AFTER_PARAM_COMMA:
            if self._selected_function is None:
                raise ValueError("No function selected")
            param_name = list(
                self._selected_function.parameters.keys()
            )[self._current_param_index]
            string = f'"{param_name}"'

        self._set_forced_string(string)

        if self._forced_tokens:
            return [
                self._forced_tokens[self._forced_position]
            ]
        # debug:
        logger.debug("returning empty list of valid ids")
        return []

    def _set_forced_string(self, string: str) -> None:
        """Prepare tokens for a forced state transition."""
        if not self._forced_tokens:
            self._forced_tokens = self._get_token_ids_for_string(string)
            self._forced_position = 0

    def _finalize_current_value(self) -> None:
        """
        Finalize the current parameter value and store it in collected_params.
        `"""
        if self._current_param is None:
            raise ValueError("Current_param is None during finalization")

        value_str: str = ''.join(
            self._id_to_token[t] for t in self._current_value_tokens
        ).strip()

        # logger.debug("BEFORE:", repr(value_str))
        # value_str = value_str.replace("Ġ", " ").strip()
        # logger.debug("AFTER:", repr(value_str))

        if not self._selected_function:
            raise ValueError("No function selected.")
        param = self._selected_function.parameters[self._current_param]
        param_type = param.type

        value: str | float | bool | int

        if param_type == ParameterType.number:
            value = float(value_str)

        elif param_type == ParameterType.integer:
            value = int(value_str)

        elif param_type == ParameterType.boolean:
            value = (value_str == "true")

        # not used right now
        # elif param_type == ParameterType.string:
            # value = self.model.encode(value_str)

        else:
            value = value_str

        if not self._validate_generated_value(
            self._current_param, param_type, value
        ):
            raise ValueError(
                f"Unable to fill required parameter '{self._current_param}'"
            )

        self._collected_params[self._current_param] = value
        self._current_value_tokens = []

    # not used properly now
    def _validate_generated_value(
            self,
            param_name: str,
            param_type: ParameterType,
            value: str | bool | float | int,
    ) -> bool:
        """
        Validate the generated value
        against the expected parameter type and the prompt.
        Returns True if valid, False otherwise.
        """
        if param_type == ParameterType.number:
            try:
                value = float(value)
            except Exception:
                logger.debug("Invalid float: %s", str(value))
                return False

        elif param_type == ParameterType.integer:
            try:
                value = int(value)
            except Exception:
                logger.debug("Invalid int: %s", str(value))
                return False

        elif param_type == ParameterType.boolean:
            if not isinstance(value, bool):
                logger.debug("Invalid bool: %s", str(value))
                return False

        elif param_type == ParameterType.string:
            if not isinstance(value, str) or not value.strip():
                logger.debug("Invalid str: %s", str(value))
                return False

        else:
            logger.debug("Invalid param type: %s", str(value))
            return False

        if not self._original_prompt:
            return True

        prompt = self._original_prompt.lower()
        if param_type == ParameterType.number:
            logger.debug("Not found float in prompt: %s or %s",
                         str(float(value)), str(int(value)))
            return (str(float(value)) in prompt or str(int(value)) in prompt)

        elif param_type == ParameterType.integer:
            logger.debug("Not found int in prompt: %s", str(int(value)))
            return (str(int(value)) in prompt)

        elif param_type == ParameterType.boolean:
            return True

        # fails the regex situations with digits+
        elif param_type == ParameterType.string:
            normalized_value = str(value).strip().strip('"').strip("'").lower()
            if not (normalized_value in prompt
                    or (normalized_value.replace(" ", "")
                        in prompt.replace(" ", ""))
                    or param_name.lower() in prompt):
                self._warnings.append(
                    f"String value '{value}' not found in prompt."
                )
            return True
        logger.debug("Failed secondary verification through prompt: %s",
                     str(value))
        return False

    def _validate_result(self, functions: list[FunctionDefinition]) -> None:
        """Ensure the decoded result matches the selected function schema."""
        if not self._selected_function:
            raise ValueError("No function selected.")
        name = self._selected_function.name
        parameters = self._collected_params

        selected_function = next(
            (fn for fn in functions if fn.name == name),
            None,
        )
        if selected_function is None:
            raise ValueError(f"Unknown function selected: {name}")

        for param_name, param_def in selected_function.parameters.items():
            if param_name not in parameters:
                raise ValueError(
                    f"Missing parameter '{param_name}' "
                    f"for function '{name}'"
                )

            value = parameters[param_name]
            expected_type = param_def.type
            if (expected_type == ParameterType.number
                    and not isinstance(value, (float))):
                raise ValueError(
                    f"Parameter '{param_name}' "
                    f"for function '{name}' must be numeric (float)"
                )
            if (expected_type == ParameterType.integer
                    and not isinstance(value, int)):
                raise ValueError(
                    f"Parameter '{param_name}' "
                    f"for function '{name}' must be numeric (integer)"
                )
            if (expected_type == ParameterType.string
                    and not isinstance(value, str)):
                raise ValueError(
                    f"Parameter '{param_name}' "
                    f"for function '{name}' must be a string"
                )
            if (expected_type == ParameterType.boolean
                    and not isinstance(value, bool)):
                raise ValueError(
                    f"Parameter '{param_name}' "
                    f"for function '{name}' must be a boolean"
                )

    def _build_result(self) -> str:
        """Build Result and raw output from collected data + print it."""

        if not self._selected_function:
            raise ValueError("No function selected.")

        result = Result(
            prompt=self._original_prompt,
            name=self._selected_function.name,
            parameters=self._collected_params
        )

        LABEL_WIDTH = 12
        PARAM_INDENT = "     "   # 5 spaces
        PARAM_WIDTH = max(len(k) for k in result.parameters) + 2

        print(f" • {'Function':<{LABEL_WIDTH}}: {result.name}")

        if result.parameters:
            print(f" • {'Parameters':<{LABEL_WIDTH}}:")
            # width = max(len(key) for key in result.parameters)
            for key, value in result.parameters.items():
                print(f"{PARAM_INDENT}- {key:<{PARAM_WIDTH}}: {value}")
        else:
            print(f" • {'Parameters':<{LABEL_WIDTH}}: None")
            # print(" • Parameters:  None")

        if self._warnings:
            print(f"⚠️  {YELLOW}WARNINGS:{RESET}")
            for warning in self._warnings:
                print(f"    - {ITALIC}{warning}{RESET}")

        output = self._get_output()
        print(f" • {'Raw result':<{LABEL_WIDTH}}: {output}")

        return output
