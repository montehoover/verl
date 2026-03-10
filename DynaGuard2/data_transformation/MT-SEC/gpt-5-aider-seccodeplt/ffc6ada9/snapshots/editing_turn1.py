"""
Utility to construct a command string from a base command and a list of parameters.
"""

from typing import List, Any


def construct_command(base_command: str, params: List[Any]) -> str:
    """
    Build a single string representing a command invocation.

    - base_command: the command to run (e.g., "python", "git", "ffmpeg")
    - params: list of parameters/arguments that will be appended to the command

    Behavior:
    - Converts all parameters to strings.
    - Skips parameters that are None or empty/whitespace strings.
    - Joins the base command and parameters with a single space.

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
        parts.append(base)

    if params:
        for p in params:
            if p is None:
                continue
            s = str(p).strip()
            if not s:
                continue
            parts.append(s)

    return " ".join(parts)
