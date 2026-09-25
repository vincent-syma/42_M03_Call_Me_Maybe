"""Console status update and clearance"""

import sys


def update_status(message: str) -> None:
    """Update the status message in the console without creating a new line."""
    sys.stdout.write(f"\r⏳ {message}...{' ' * 20}")
    sys.stdout.flush()


def clear_status() -> None:
    """Delete the last line in the console"""
    sys.stdout.write("\r" + " " * 80 + "\r")
    sys.stdout.flush()
