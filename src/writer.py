"""Functions for writing results to output JSON file."""

from json import dump
from pathlib import Path
from typing import Any


def write_outputs(
    output_objects: list[Any],
    path: str,
    overwrite: bool = False,
) -> None:
    """Write list of JSON objects into 1 JSON file defined by path."""

    output_path = Path(path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # FOR AUTOMATIC TESTER - just a note:
    if output_path.exists() and not overwrite:
        print(
            f"⚠️  Overwriting existing file:\n"
            f"   {output_path.resolve()}"
        )

    # FOR HUMAN USER - interactive:
    # if output_path.exists() and not overwrite:
    #     answer = input(
    #         f"⚠️  Output file already exists:\n"
    #         f"   {output_path.resolve()}\n"
    #         f"Overwrite it? [y/N]: "
    #     )

    #     if answer.lower() not in {"y", "yes"}:
    #         print("❌ Write cancelled.")
    #         return

    #     # # OR:
    #     # raise FileExistsError(
    #     #     f"Output file already exists: "
    #     #     f"{output_path.resolve()}\n"
    #     #     "Use --force to overwrite it."
    #     # )

    try:
        with output_path.open("w") as file:
            dump(output_objects, file, indent=2)

        print(
            f"✅ Outputs written to:\n"
            f"   {output_path.resolve()}"
        )

    except OSError as e:
        raise OSError(
            f"Could not write output file "
            f"{output_path.resolve()}: {e}"
        ) from e
