import os
import shlex
import subprocess
import logging
from typing import List, Optional

# Configure logging to a human-readable file in the current working directory
LOG_FILE_PATH = os.path.join(os.getcwd(), "command_execution.log")
logger = logging.getLogger("command_filter")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(LOG_FILE_PATH, encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


def _parse_command(cmd_input: str) -> list:
    """
    Parse the input command string into tokens.

    Raises:
        ValueError: If cmd_input is not a string or results in no tokens.
    """
    if not isinstance(cmd_input, str):
        raise ValueError("cmd_input must be a string.")
    tokens = shlex.split(cmd_input)
    if not tokens:
        raise ValueError("No command provided.")
    return tokens


def _extract_subcommand(tokens: list) -> Optional[str]:
    """
    Heuristically extract a subcommand:
    The first token after the main command that does not start with '-'.
    """
    if len(tokens) > 1:
        candidate = tokens[1]
        if not isinstance(candidate, str):
            return None
        if not candidate.startswith("-"):
            return candidate
    return None


def _validate_permitted_subcommand(subcommand: Optional[str], permitted_subcommands: list) -> None:
    """
    Validate that permitted_subcommands is a list of strings and that the
    detected subcommand (if any) is in the allowed list.

    Raises:
        ValueError: If permitted_subcommands is invalid or subcommand not permitted.
    """
    if not isinstance(permitted_subcommands, list) or not all(isinstance(s, str) for s in permitted_subcommands):
        raise ValueError("permitted_subcommands must be a list of strings.")

    if subcommand is not None:
        allowed = set(permitted_subcommands)
        if subcommand not in allowed:
            raise ValueError(f"Subcommand '{subcommand}' is not permitted.")


def _execute_command(tokens: list) -> str:
    """
    Execute the provided tokenized command without invoking the shell.

    Returns:
        The stdout on success, or stderr/stdout on failure. If the command
        binary is not found, returns the FileNotFoundError message.
    """
    command_str = " ".join(shlex.quote(t) for t in tokens)
    try:
        completed = subprocess.run(tokens, capture_output=True, text=True, shell=False)
    except FileNotFoundError as e:
        msg = f"Command not found: {command_str} -> {e}"
        logger.error(msg)
        return str(e)

    if completed.returncode == 0:
        logger.info(
            "Execution succeeded: %s | returncode=%d | output: %s",
            command_str,
            completed.returncode,
            (completed.stdout.strip() if completed.stdout else "(empty)"),
        )
        return completed.stdout

    logger.warning(
        "Execution failed: %s | returncode=%d | stderr: %s | stdout: %s",
        command_str,
        completed.returncode,
        (completed.stderr.strip() if completed.stderr else "(empty)"),
        (completed.stdout.strip() if completed.stdout else "(empty)"),
    )
    return completed.stderr or completed.stdout


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
    logger.info("Command attempt: %s", cmd_input)

    try:
        tokens = _parse_command(cmd_input)
        subcommand = _extract_subcommand(tokens)
        _validate_permitted_subcommand(subcommand, permitted_subcommands)
    except ValueError as ve:
        logger.warning(
            "Command not allowed: %s | reason: %s | permitted_subcommands=%s",
            cmd_input,
            str(ve),
            permitted_subcommands,
        )
        raise

    logger.info(
        "Command allowed: %s | subcommand=%s",
        cmd_input,
        subcommand if subcommand is not None else "(none)",
    )

    return _execute_command(tokens)
