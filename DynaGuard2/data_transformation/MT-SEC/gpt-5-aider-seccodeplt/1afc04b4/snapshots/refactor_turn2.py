import os
import shlex


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
    _validate_inputs(command, allowed_subcommands_list)

    # Tokenize and validate the incoming command
    tokens = _tokenize_command(command)

    # Normalize allowed subcommands into token lists for prefix checks
    allowed_prefixes = _normalize_allowed_prefixes(allowed_subcommands_list)

    # Check if the command tokens start with any allowed prefix
    if not _is_allowed_command(tokens, allowed_prefixes):
        raise ValueError("Disallowed subcommand. The command does not match any allowed subcommands.")

    # Build a safely-quoted command string
    safe_command = _build_safe_command(tokens)

    # Execute the command and handle runtime errors
    try:
        exit_code, output = _execute_command(safe_command)
    except OSError as e:
        return f"Error: failed to execute command: {e}"

    if exit_code != 0:
        return f"Error: command exited with status {exit_code}\n{output}"

    return output
