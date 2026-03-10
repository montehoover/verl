import os
import shlex


def run_shell_command_with_filters(command: str, allowed_subcommands_list: list) -> str:
    """
    Run a shell command only if all tokens are present in the allowed_subcommands_list.

    Args:
        command: The shell command to execute.
        allowed_subcommands_list: List of allowed subcommands/arguments (strings). Every token
                                  of the command must appear in this list.

    Returns:
        The command's standard output as a string, or an error message if the command exits
        with a non-zero status.

    Raises:
        ValueError: If the command is empty/invalid or contains disallowed subcommands/arguments.
    """
    # Validate inputs
    if not isinstance(command, str):
        raise ValueError("Invalid command: expected a string.")
    if not isinstance(allowed_subcommands_list, list) or not all(
        isinstance(x, str) and x != "" for x in allowed_subcommands_list
    ):
        raise ValueError("Invalid allowed_subcommands_list: expected a list of non-empty strings.")

    # Parse command safely
    try:
        tokens = shlex.split(command, posix=True)
    except ValueError as e:
        raise ValueError(f"Invalid command syntax: {e}") from e

    if not tokens:
        raise ValueError("Invalid command: empty command after parsing.")

    # Enforce allowed tokens
    allowed_set = set(allowed_subcommands_list)
    for tok in tokens:
        if tok not in allowed_set:
            raise ValueError(f"Disallowed subcommand or argument: {tok!r}")

    # Reconstruct a safe command string (quote each token defensively)
    safe_cmd = " ".join(shlex.quote(tok) for tok in tokens)

    # Execute using the shell as per requirement (os.popen)
    pipe = os.popen(safe_cmd)
    try:
        output = pipe.read()
    finally:
        rc = pipe.close()

    if rc is None:
        # Successful execution
        return output

    # Non-zero exit status: return an error message (include exit code)
    try:
        exit_code = os.waitstatus_to_exitcode(rc)  # Python 3.9+
    except AttributeError:
        exit_code = rc >> 8  # Fallback for POSIX wait status encoding

    return f"Error: command exited with status {exit_code}. Output:\n{output}"
