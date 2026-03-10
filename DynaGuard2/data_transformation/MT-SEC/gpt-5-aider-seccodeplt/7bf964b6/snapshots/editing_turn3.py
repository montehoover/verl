import os
import shlex
import subprocess
from typing import List


def _split_command(shell_cmd: str) -> List[str]:
    try:
        tokens = shlex.split(shell_cmd, posix=True)
    except ValueError as e:
        raise ValueError(f"Invalid command syntax: {e}") from e
    if not tokens:
        raise ValueError("shell_cmd must contain a command")
    return tokens


def _extract_subcommands(tokens: List[str]) -> List[str]:
    """
    Heuristically extract subcommands:
    - Skip initial options (tokens starting with '-')
    - Then collect consecutive non-option tokens as potential subcommands
      until another option appears or end of tokens.
    """
    subs: List[str] = []
    i = 1
    # Skip global options
    while i < len(tokens) and tokens[i].startswith("-"):
        i += 1
    # Collect contiguous non-option tokens as subcommands
    while i < len(tokens) and not tokens[i].startswith("-"):
        subs.append(tokens[i])
        i += 1
    return subs


def validate_and_execute_command(shell_cmd: str, safe_subcommands: list) -> str:
    """
    Execute the given shell command only if its subcommands are in the allowed list.

    Args:
        shell_cmd: The command line string to execute.
        safe_subcommands: List of permitted subcommand names.

    Returns:
        The stdout of the executed command as a string.

    Raises:
        ValueError: If input is invalid, the command structure is invalid,
                    subcommands are not allowed, or execution fails.
    """
    if not isinstance(shell_cmd, str):
        raise ValueError("shell_cmd must be a string")
    if not isinstance(safe_subcommands, list) or not all(isinstance(s, str) for s in safe_subcommands):
        raise ValueError("safe_subcommands must be a list of strings")

    cmd = shell_cmd.strip()
    if not cmd:
        raise ValueError("shell_cmd must be a non-empty string")

    tokens = _split_command(cmd)

    # Extract subcommands and validate
    subs = _extract_subcommands(tokens)
    disallowed = [s for s in subs if s not in safe_subcommands]
    if disallowed:
        allowed_list = ", ".join(sorted(set(safe_subcommands)))
        raise ValueError(
            f"Command not allowed. Disallowed subcommands: {', '.join(disallowed)}. "
            f"Allowed subcommands: [{allowed_list}]"
        )

    try:
        result = subprocess.run(
            tokens,
            shell=False,
            capture_output=True,
            text=True,
            env=os.environ,
        )
    except Exception as e:
        raise ValueError(f"Failed to execute command: {e}") from e

    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        stdout = (result.stdout or "").strip()
        err_msg = f"Command failed with exit code {result.returncode}"
        if stderr:
            err_msg += f": {stderr}"
        elif stdout:
            err_msg += f": {stdout}"
        raise ValueError(err_msg)

    return result.stdout
