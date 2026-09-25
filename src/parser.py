"""Parser for handling command line arguments."""

import argparse


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(description="Function calling tool")
    parser.add_argument(
        '--functions_definition',
        default='data/input/functions_definition.json'
    )
    parser.add_argument(
        '--input',
        default='data/input/function_calling_tests.json'
    )
    parser.add_argument(
        '--output',
        default='data/output/function_calls.json'
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="overwrite existing output file",
    )
    # parser.add_argument(
    #     "--model",
    #     help="model package",
    # )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
    )
    return parser.parse_args()
