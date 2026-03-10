"""
Utility to construct a command string from a base command and a list of parameters.
"""

import subprocess
import shlex
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


def execute_command(base_command: str, parameters: List[Any]) -> str:
    """
    Construct and execute a command.

    Arguments:
        base_command: The command to execute (e.g., "python", "git").
        parameters: A list of parameters/arguments to pass to the command.

    Returns:
        The standard output of the command as a string.

    Raises:
        ValueError: If the command is empty, contains invalid characters,
                    cannot be found, or exits with a non-zero status.
    """
    base = (str(base_command).strip() if base_command is not None else "")

    tokens: List[str] = []
    if base:
        _validate_token(base, "base_command")
        tokens.append(base)

    if parameters:
        for idx, p in enumerate(parameters):
            if p is None:
                continue
            s = str(p).strip()
            if not s:
                continue
            _validate_token(s, f"parameters[{idx}]")
            tokens.append(s)

    if not tokens:
        raise ValueError("Constructed command is empty (no valid base command or parameters).")

    # Shell-escaped string for diagnostics/logging
    command_str = shlex.join(tokens)

    try:
        result = subprocess.run(
            tokens,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        stderr = e.stderr or ""
        raise ValueError(f"Command failed with exit code {e.returncode}: {command_str}\n{stderr}") from None
    except FileNotFoundError:
        raise ValueError(f"Command not found: {tokens[0]}") from None
    except OSError as e:
        raise ValueError(f"Failed to execute command: {command_str}\n{e}") from None
