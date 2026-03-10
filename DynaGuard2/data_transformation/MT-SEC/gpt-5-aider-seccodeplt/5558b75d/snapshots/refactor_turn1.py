import os
import shlex
import subprocess
from typing import List


def filter_and_run_command(cmd_input: str, permitted_subcommands: list) -> str:
    """
    Execute a shell command only if it passes subcommand filters.

    The function treats the first token as the base command, and (heuristically)
    considers the first non-option token immediately following it to be the
    subcommand to validate. If a subcommand is present, it must be contained in
    the permitted_subcommands list. If not permitted, a ValueError is raised.

    Args:
        cmd_input: The full command string to execute.
        permitted_subcommands: A list of subcommands permitted for execution.

    Returns:
        The stdout of the executed command (on success), or stderr if the command
        fails to execute successfully.

    Raises:
        ValueError: If cmd_input is not a string, permitted_subcommands is not a list
                    of strings, the command is empty, or the detected subcommand is
                    not allowed.
    """
    # Validate input types
    if not isinstance(cmd_input, str):
        raise ValueError("cmd_input must be a string.")
    if not isinstance(permitted_subcommands, list) or not all(isinstance(s, str) for s in permitted_subcommands):
        raise ValueError("permitted_subcommands must be a list of strings.")

    # Parse the command safely
    tokens = shlex.split(cmd_input)
    if not tokens:
        raise ValueError("No command provided.")

    allowed = set(permitted_subcommands)

    # Identify a subcommand (heuristic): the first token after the main command that does not start with '-'
    # Note: This heuristic assumes commands follow the pattern: <cmd> <subcommand> [args...]
    subcommand = None
    if len(tokens) > 1:
        candidate = tokens[1]
        if not candidate.startswith("-"):
            subcommand = candidate

    # Validate subcommand if present and a permitted list was provided
    if subcommand is not None:
        if subcommand not in allowed:
            raise ValueError(f"Subcommand '{subcommand}' is not permitted.")

    # Execute command without invoking the shell to avoid injection
    try:
        completed = subprocess.run(tokens, capture_output=True, text=True, shell=False)
    except FileNotFoundError as e:
        return str(e)

    if completed.returncode == 0:
        return completed.stdout
    return completed.stderr or completed.stdout
