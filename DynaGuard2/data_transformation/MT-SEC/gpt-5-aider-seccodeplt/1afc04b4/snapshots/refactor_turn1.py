import os
import shlex


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
    # Basic validations
    if not isinstance(command, str):
        raise ValueError("Command must be a string.")
    if not isinstance(allowed_subcommands_list, list):
        raise ValueError("allowed_subcommands_list must be a list of strings.")
    if not allowed_subcommands_list:
        raise ValueError("allowed_subcommands_list cannot be empty.")
    if any(not isinstance(item, str) or not item.strip() for item in allowed_subcommands_list):
        raise ValueError("allowed_subcommands_list must contain non-empty strings only.")

    # Tokenize the incoming command safely
    try:
        tokens = shlex.split(command, posix=True)
    except ValueError as e:
        raise ValueError(f"Invalid command syntax: {e}") from e

    if not tokens:
        raise ValueError("Command cannot be empty.")

    # Normalize allowed subcommands into token lists for prefix checks
    allowed_prefixes = []
    for entry in allowed_subcommands_list:
        try:
            prefix_tokens = shlex.split(entry, posix=True)
        except ValueError as e:
            raise ValueError(f"Invalid allowed subcommand syntax '{entry}': {e}") from e
        if not prefix_tokens:
            raise ValueError("Allowed subcommand entries cannot be empty.")
        allowed_prefixes.append(prefix_tokens)

    # Check if the command tokens start with any allowed prefix
    is_allowed = any(tokens[:len(prefix)] == prefix for prefix in allowed_prefixes)
    if not is_allowed:
        raise ValueError("Disallowed subcommand. The command does not match any allowed subcommands.")

    # Re-quote each token to neutralize shell metacharacters before passing to the shell
    safe_command = " ".join(shlex.quote(tok) for tok in tokens)

    try:
        stream = os.popen(safe_command, "r")
        output = stream.read()
        status = stream.close()  # None means success (exit code 0) on POSIX
    except OSError as e:
        return f"Error: failed to execute command: {e}"

    if status not in (None, 0):
        # Normalize exit status across platforms
        exit_code = None
        try:
            exit_code = os.waitstatus_to_exitcode(status)  # POSIX returns wait status
        except Exception:
            exit_code = status if isinstance(status, int) else 1
        return f"Error: command exited with status {exit_code}\n{output}"

    return output
