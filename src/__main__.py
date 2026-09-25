#!/usr/bin/env python3
"""Entry point for the Call Me Maybe function calling tool."""

import sys
from rich import print
from .format_rich import BOLD_RED, RESET
from typing import Any

from .parser import parse_args
from .loader import load_function_definitions, load_test_prompts
from .decoder import Decoder
from .writer import write_outputs
# from .status import clear_status

# from .models import LLMModel
# from llm_sdk import Small_LLM_Model as LLM

import time
import logging

logger = logging.getLogger(__name__)


def main() -> int:
    """Entry point of the Call Me Maybe program"""

    start = time.time()

    # try:
    #     model = LLM()
    #     # # device="cpu": to avoid CUDA mismatch of torch and device:
    #     # model = LLM(device="cpu")
    # except Exception as e:
    #     print(f"❌ {BOLD_RED}ERROR:{RESET} Failed to load LLM: {e}")
    #     return 1

    try:
        # parse and load CLI arguments
        args = parse_args()

        # debug logs
        logging.basicConfig(
            level=logging.WARNING,
            format="%(levelname)s: %(name)s:\t%(message)s",
        )

        if args.debug:
            logging.getLogger("src.decoder").setLevel(logging.DEBUG)
            logging.getLogger("src.__main__").setLevel(logging.DEBUG) # not tested

        function_defs = load_function_definitions(args.functions_definition)
        # debug:
        for fn in function_defs:
            logger.debug("- %s", fn.name)

        test_prompts = load_test_prompts(args.input)

        # if args.model:
        #     decoder = Decoder(
        #         model=LLMModel(
        #             backend=args.model
        #         )
        #     )
        # else:
        #     decoder = Decoder()

        decoder = Decoder()

        outputs: list[dict[str, Any]] = []

        print()
        print('-' * 50)

        # print("Using just 'select_function' method:\n"
        #       f"{decoder.decode(test_prompts[0].prompt, function_defs)}")

        for i, test_prompt in enumerate(test_prompts):
            prompt_start = time.time()
            # print(f"\nProcessing prompt {i + 1}/{len(test_prompts)}..."
            LABEL_WIDTH = 12
            label = f"Prompt {i + 1}/{len(test_prompts)}"
            print(f"\n • {label:<{LABEL_WIDTH}}: '{test_prompt.prompt}'")

            output = decoder.decode(function_defs, test_prompt.prompt)
            outputs.append(output)
            prompt_elapsed = time.time() - prompt_start
            print(f"\nPrompt processed in:   {prompt_elapsed:.4f}s\n")
            print('-' * 50)

            # clear_status() # delete the status line after each prompt
            # print("✅ Done!")  # Chosen function: {BOLD}{result.name}{RESET}")

        print()
        write_outputs(outputs, args.output, overwrite=False)
        total_elapsed = time.time() - start
        print(f"\n{len(test_prompts)} prompts proccessed in:"
              f"   {total_elapsed:.4f}s")
        print('-' * 50)
        return 0

    except (
        FileNotFoundError,
        PermissionError,
        ValueError,
        OSError
     ) as e:
        print(f"❌ {BOLD_RED}ERROR:{RESET} {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
