import os
import shlex
from typing import List


def filter_and_run_command(cmd_input: str, permitted_subcommands: List[str]) -> str:
    """
    Execute a shell command only if every token (command and its subcommands/flags)
    is present in the permitted_subcommands list.

    Args:
        cmd_input: The command string to execute (e.g., "ls -l /tmp").
        permitted_subcommands: List of permitted tokens (e.g., ["ls", "-l", "/tmp"]).

    Returns:
        The stdout output of the executed command as a string.

    Raises:
        ValueError: If cmd_input is empty, not a string, permitted_subcommands is invalid,
                    or any token in the parsed command is not in the permitted list.
    """
    if not isinstance(cmd_input, str):
        raise ValueError("cmd_input must be a string.")
    if not isinstance(permitted_subcommands, list) or not all(isinstance(s, str) for s in permitted_subcommands):
        raise ValueError("permitted_subcommands must be a list of strings.")

    tokens = shlex.split(cmd_input)
    if not tokens:
        raise ValueError("No command provided.")

    permitted_set = set(permitted_subcommands)
    disallowed = [tok for tok in tokens if tok not in permitted_set]
    if disallowed:
        # Raise with a clear error message indicating which tokens are not allowed
        raise ValueError(f"Command not allowed. Disallowed tokens: {', '.join(disallowed)}")

    # Reconstruct a safe command string using shlex.quote to prevent injection
    safe_cmd = " ".join(shlex.quote(tok) for tok in tokens)

    # Execute the command and return its output
    stream = os.popen(safe_cmd)
    try:
        output = stream.read()
    finally:
        stream.close()
    return output
