"""
Utility to construct a command string from a base command and a list of parameters.
"""

from typing import List, Any


def _validate_token(token: str, context: str) -> None:
    """
    Validate a single command token.

    Rules:
    - Must be non-empty.
    - Must be printable (no control or non-printable characters).
    """
    if not token:
        raise ValueError(f"Empty token encountered while validating {context}")
    if not token.isprintable():
        raise ValueError(f"Token contains non-printable characters in {context}: {repr(token)}")


def construct_command(base_command: str, params: List[Any]) -> str:
    """
    Build a single string representing a command invocation.

    - base_command: the command to run (e.g., "python", "git", "ffmpeg")
    - params: list of parameters/arguments that will be appended to the command

    Behavior:
    - Converts all parameters to strings.
    - Skips parameters that are None or empty/whitespace strings.
    - Joins the base command and parameters with a single space.
    - Validates that all tokens are printable and the final command is non-empty.

    Raises:
        ValueError: If the constructed command is empty or any token contains
                    non-printable/control characters.

    Examples:
        construct_command("python", ["-m", "http.server", 8000])
        -> "python -m http.server 8000"

        construct_command("git", ["commit", "-m", "Initial commit"])
        -> "git commit -m Initial commit"

        construct_command("ffmpeg", None)  # type: ignore
        -> "ffmpeg"

        construct_command("", ["--help"])
        -> "--help"
    """
    base = (str(base_command).strip() if base_command is not None else "")

    parts: List[str] = []
    if base:
        _validate_token(base, "base_command")
        parts.append(base)

    if params:
        for idx, p in enumerate(params):
            if p is None:
                continue
            s = str(p).strip()
            if not s:
                continue
            _validate_token(s, f"params[{idx}]")
            parts.append(s)

    if not parts:
        raise ValueError("Constructed command is empty (no valid base command or parameters).")

    command = " ".join(parts)

    # Final sanity check on the resulting command string
    if not command.isprintable():
        raise ValueError("Constructed command contains non-printable characters.")

    return command
