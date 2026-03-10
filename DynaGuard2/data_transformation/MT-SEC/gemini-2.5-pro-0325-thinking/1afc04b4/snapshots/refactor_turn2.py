import os
import shlex


def _validate_command(command: str, allowed_subcommands_list: list) -> None:
    """
    Validates the command string and checks if its subcommand is allowed.

    Args:
        command: The shell command string.
        allowed_subcommands_list: A list of allowed subcommands.

    Raises:
        ValueError: If the command is invalid or the subcommand is not allowed.
    """
    if not command:
        raise ValueError("Command cannot be empty.")

    try:
        parts = shlex.split(command)
    except ValueError as e:  # shlex.split can raise ValueError for unclosed quotes
        raise ValueError(f"Invalid command syntax: {e}") from e

    if not parts:  # shlex.split on " " or "" can result in an empty list
        raise ValueError("Command cannot be empty after parsing.")

    # Expects command structure: <main_command> <sub_command> [args...]
    # parts[0] is main_command, parts[1] is sub_command.
    if len(parts) < 2:
        raise ValueError(
            f"Command '{command}' does not include a subcommand. "
            "Expected structure: <command> <subcommand> [options]"
        )

    subcommand_to_check = parts[1]

    if subcommand_to_check not in allowed_subcommands_list:
        raise ValueError(
            f"Subcommand '{subcommand_to_check}' is not allowed. "
            f"Allowed subcommands are: {allowed_subcommands_list}"
        )


def _execute_command(command: str) -> tuple[str, int | None]:
    """
    Executes the given shell command using os.popen.

    Args:
        command: The shell command string to execute.

    Returns:
        A tuple containing:
            - output (str): The standard output of the command.
            - exit_status_raw (int | None): The raw exit status from pipe.close().
                                            This can be None if pipe.close() itself fails or
                                            if an exception occurred before it could be set meaningfully.

    Raises:
        OSError: or other exceptions from os.popen or file operations if they occur.
    """
    pipe_obj = None
    output = ""
    exit_status_raw = None  # Initialize to ensure it's always defined

    try:
        pipe_obj = os.popen(command)
        output = pipe_obj.read()  # Reads stdout
    finally:
        if pipe_obj:
            # This close() is crucial. It also sets the exit status.
            # If os.popen() or read() fails, this finally block still runs.
            exit_status_raw = pipe_obj.close()
            # If close() itself raises an error, that error will propagate.

    return output, exit_status_raw


def run_shell_command_with_filters(command: str, allowed_subcommands_list: list) -> str:
    """
    Runs a shell command if its subcommand is in the allowed list.

    Args:
        command: The shell command string to execute (e.g., "utility subcommand --option value").
        allowed_subcommands_list: A list of strings representing allowed subcommands.
                                  The subcommand is expected to be the token following the main command.

    Returns:
        The standard output of the command if successful, or an error message string if the
        command fails to execute or returns a non-zero exit status.

    Raises:
        ValueError: If the command string is empty, syntactically invalid,
                    does not include a subcommand, or if the subcommand is not in
                    the allowed_subcommands_list.
    """
    # Validate the command and subcommand
    _validate_command(command, allowed_subcommands_list)

    output: str
    exit_status_raw: int | None

    try:
        output, exit_status_raw = _execute_command(command)
    except Exception as e:
        # This catches errors from _execute_command (e.g., os.popen, read, or close failures)
        return f"Failed during command execution or output reading for '{command}': {e}"

    # Determine if there was an error based on the exit status
    is_error = False
    actual_exit_status_str = "0"  # Default for success message or if no status

    if exit_status_raw is not None:
        # POSIX: exit_status_raw is None for success (0), or encoded status for non-zero/signal.
        # Windows: exit_status_raw is the integer exit code (0 for success).
        if hasattr(os, 'WIFEXITED') and os.WIFEXITED(exit_status_raw):  # POSIX normal exit
            status_code = os.WEXITSTATUS(exit_status_raw)
            actual_exit_status_str = str(status_code)
            if status_code != 0:
                is_error = True
        elif hasattr(os, 'WIFSIGNALED') and os.WIFSIGNALED(exit_status_raw):  # POSIX signal termination
            actual_exit_status_str = f"terminated by signal {os.WTERMSIG(exit_status_raw)}"
            is_error = True # Signals are errors
        elif not hasattr(os, 'WIFEXITED'): # Likely Windows or other non-POSIX
            status_code = exit_status_raw # On Windows, this is the direct exit code
            actual_exit_status_str = str(status_code)
            if status_code != 0:
                is_error = True
        else: # Unhandled case for exit_status_raw, treat as error if non-zero-like
            actual_exit_status_str = str(exit_status_raw) # Raw status
            is_error = True # Default to error if unknown structure but not None
    # If exit_status_raw is None (POSIX success, exit code 0), is_error remains False.

    if is_error:
        error_detail = output.strip() if output.strip() else "No output on stdout. Error details may be on stderr (not captured by this method)."
        return (
            f"Error executing command: '{command}'. "
            f"Exit status: {actual_exit_status_str}. "
            f"Output (stdout): {error_detail}"
        )

    return output.strip()  # Success
