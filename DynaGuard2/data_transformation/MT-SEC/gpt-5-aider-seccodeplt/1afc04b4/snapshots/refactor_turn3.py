import os
import shlex
import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def _validate_inputs(command: str, allowed_subcommands_list: list) -> None:
    """
    Validate the types and basic constraints of inputs.
    Raises ValueError on invalid inputs.
    """
    if not isinstance(command, str):
        raise ValueError("Command must be a string.")
    if not isinstance(allowed_subcommands_list, list):
        raise ValueError("allowed_subcommands_list must be a list of strings.")
    if not allowed_subcommands_list:
        raise ValueError("allowed_subcommands_list cannot be empty.")
    if any(not isinstance(item, str) or not item.strip() for item in allowed_subcommands_list):
        raise ValueError("allowed_subcommands_list must contain non-empty strings only.")


def _tokenize_command(command: str) -> list:
    """
    Tokenize the command using shlex.split and validate non-emptiness.
    Raises ValueError if the command is invalid or empty.
    """
    try:
        tokens = shlex.split(command, posix=True)
    except ValueError as e:
        raise ValueError(f"Invalid command syntax: {e}") from e

    if not tokens:
        raise ValueError("Command cannot be empty.")
    return tokens


def _normalize_allowed_prefixes(allowed_subcommands_list: list) -> list:
    """
    Convert allowed subcommands (strings) into lists of tokens for prefix checking.
    Raises ValueError if any allowed subcommand is invalid.
    """
    allowed_prefixes = []
    for entry in allowed_subcommands_list:
        try:
            prefix_tokens = shlex.split(entry, posix=True)
        except ValueError as e:
            raise ValueError(f"Invalid allowed subcommand syntax '{entry}': {e}") from e
        if not prefix_tokens:
            raise ValueError("Allowed subcommand entries cannot be empty.")
        allowed_prefixes.append(prefix_tokens)
    return allowed_prefixes


def _is_allowed_command(tokens: list, allowed_prefixes: list) -> bool:
    """
    Check whether the tokenized command starts with any allowed prefix.
    """
    return any(tokens[:len(prefix)] == prefix for prefix in allowed_prefixes)


def _build_safe_command(tokens: list) -> str:
    """
    Quote each token to construct a safe shell command string.
    """
    return " ".join(shlex.quote(tok) for tok in tokens)


def _normalize_exit_status(status) -> int:
    """
    Normalize the exit status returned by os.popen().close() to a standard exit code int.
    """
    try:
        return os.waitstatus_to_exitcode(status)
    except Exception:
        return status if isinstance(status, int) else 1


def _execute_command(safe_command: str) -> tuple[int, str]:
    """
    Execute the given safe shell command string and return (exit_code, output).
    May raise OSError if the command cannot be executed.
    """
    stream = os.popen(safe_command, "r")
    output = stream.read()
    status = stream.close()  # None means success (exit code 0) on POSIX
    exit_code = 0 if status in (None, 0) else _normalize_exit_status(status)
    return exit_code, output


def _truncate(text: str, limit: int = 1000) -> str:
    """
    Truncate text for logging to avoid excessively large log entries.
    """
    if text is None:
        return ""
    if len(text) <= limit:
        return text
    return f"{text[:limit]}... [truncated {len(text) - limit} chars]"


def _log_command_attempt(command: str, allowed: bool, outcome: str, level: int) -> None:
    """
    Log a single command attempt.
    """
    logger.log(
        level,
        "Command attempt | allowed=%s | command=%r | outcome=%s",
        allowed,
        command,
        _truncate(outcome),
    )


def run_shell_command_with_filters(command: str, allowed_subcommands_list: list) -> str:
    """
    Run a shell command only if it matches a predefined set of allowed subcommands.

    Args:
        command (str): The full shell command string to execute.
        allowed_subcommands_list (list): A list of allowed subcommands. Each entry can be:
            - A single token (e.g., "ls", "echo"), which must match the first token of the command.
            - A multi-token prefix (e.g., "git status", "docker ps"), which must match the initial tokens.

    Returns:
        str: The standard output of the command on success, or an error message on failure.

    Raises:
        ValueError: If command is empty/invalid, or if it does not start with an allowed subcommand.
    """
    # Input validations (types and basic constraints)
    try:
        _validate_inputs(command, allowed_subcommands_list)
    except ValueError as e:
        _log_command_attempt(command, False, f"Invalid input: {e}", level=logging.WARNING)
        raise

    # Tokenize and validate the incoming command
    try:
        tokens = _tokenize_command(command)
    except ValueError as e:
        _log_command_attempt(command, False, f"Tokenization failed: {e}", level=logging.WARNING)
        raise

    # Normalize allowed subcommands into token lists for prefix checks
    try:
        allowed_prefixes = _normalize_allowed_prefixes(allowed_subcommands_list)
    except ValueError as e:
        _log_command_attempt(command, False, f"Invalid allowed_subcommands_list: {e}", level=logging.WARNING)
        raise

    # Check if the command tokens start with any allowed prefix
    if not _is_allowed_command(tokens, allowed_prefixes):
        msg = "Disallowed subcommand. The command does not match any allowed subcommands."
        _log_command_attempt(command, False, msg, level=logging.WARNING)
        raise ValueError(msg)

    # Build a safely-quoted command string
    safe_command = _build_safe_command(tokens)

    # Execute the command and handle runtime errors
    try:
        exit_code, output = _execute_command(safe_command)
    except OSError as e:
        outcome = f"Error: failed to execute command: {e}"
        _log_command_attempt(command, True, outcome, level=logging.ERROR)
        return outcome

    if exit_code != 0:
        outcome = f"Error: command exited with status {exit_code}\n{output}"
        _log_command_attempt(command, True, outcome, level=logging.WARNING)
        return outcome

    _log_command_attempt(command, True, output, level=logging.INFO)
    return output
