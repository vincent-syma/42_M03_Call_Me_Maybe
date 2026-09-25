"""Functions for loading and validating input JSON files."""

from pydantic import ValidationError
from json import JSONDecodeError, load
from .models import FunctionDefinition, TestPrompt


def load_function_definitions(path: str) -> list[FunctionDefinition]:
    """Store function definitions from the input file
    into list of pydantic validated models"""

    try:
        with open(path) as file:
            data = load(file)

    except FileNotFoundError:
        raise FileNotFoundError(f"Function definitions file {path} not found.")

    except PermissionError:
        raise PermissionError(f"File {path} not permitted to read.")

    except JSONDecodeError as e:
        raise ValueError(f"Invalid JSON file: {path}: {e}")

    result = []

    for item in data:
        try:
            result.append(FunctionDefinition.model_validate(item))

        except ValidationError as e:
            raise ValueError(
                f"Invalid function definition/s in {path}: {e}"
            )
    return result


def load_test_prompts(path: str) -> list[TestPrompt]:
    """Stores test prompts from the input file
    into list of pydantic validated models"""

    try:
        with open(path) as file:
            data = load(file)

    except FileNotFoundError:
        raise FileNotFoundError(f"Test prompts file {path} not found.")

    except PermissionError:
        raise PermissionError(f"File {path} not permitted to read.")

    except JSONDecodeError as e:
        raise ValueError(f"Invalid JSON file: {path}: {e}")

    result = []

    for item in data:
        try:
            result.append(TestPrompt.model_validate(item))

        except ValidationError as e:
            raise ValueError(
                f"Invalid test prompt/s in {path}: {e}"
            )
    return result
