import os
import shlex

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
    if not command:
        raise ValueError("Command cannot be empty.")

    try:
        parts = shlex.split(command)
    except ValueError as e: # shlex.split can raise ValueError for unclosed quotes
        raise ValueError(f"Invalid command syntax: {e}") from e

    if not parts: # shlex.split on " " or "" can result in an empty list
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

    # Execute the command using os.popen
    pipe_obj = None
    output = ""
    exit_status_raw = None # Represents the raw value from pipe.close()

    try:
        pipe_obj = os.popen(command)
        output = pipe_obj.read() # Reads stdout
        # stderr is not directly captured by os.popen into this stream.
    except Exception as e:
        # This catches errors during the os.popen() call itself or pipe_obj.read()
        if pipe_obj:
            # Attempt to close the pipe if it was opened, though its status might be lost/irrelevant
            # if the error was in popen() itself or read().
            try:
                pipe_obj.close()
            except Exception:
                pass # Ignore errors during close in this exceptional path
        return f"Failed during command execution or output reading for '{command}': {e}"
    finally:
        if pipe_obj:
            # This is the primary path for getting the exit status.
            # If read() failed, pipe_obj.close() might be called here if not in except block.
            # Ensure it's only called once if possible, or that subsequent calls are safe.
            # os.popen objects' close() can typically be called multiple times, returning None on subsequent calls.
            # However, to be safe, we get status here if not already closed due to an earlier error.
            if exit_status_raw is None: # Check if not already closed and status obtained (e.g. if read failed)
                 exit_status_raw = pipe_obj.close()


    if exit_status_raw is not None: # Command execution failed or returned non-zero
        actual_exit_status = 'unknown'
        # Check if the command exited normally (os.WIFEXITED is POSIX specific)
        if hasattr(os, 'WIFEXITED') and os.WIFEXITED(exit_status_raw):
            actual_exit_status = os.WEXITSTATUS(exit_status_raw)
        # Check if the command was terminated by a signal (os.WIFSIGNALED is POSIX specific)
        elif hasattr(os, 'WIFSIGNALED') and os.WIFSIGNALED(exit_status_raw):
            actual_exit_status = f"terminated by signal {os.WTERMSIG(exit_status_raw)}"
        else:
            # For non-POSIX or if status is not from exit/signal, use raw status.
            # On Windows, exit_status_raw is directly the exit code.
            actual_exit_status = exit_status_raw >> 8 if hasattr(os, 'WIFEXITED') else exit_status_raw


        error_detail = output.strip() if output.strip() else "No output on stdout. Error details may be on stderr (not captured by this method)."
        return (
            f"Error executing command: '{command}'. "
            f"Exit status: {actual_exit_status}. "
            f"Output (stdout): {error_detail}"
        )
    
    return output.strip() # Success
