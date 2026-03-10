import os
import shlex

def execute_command(command: str) -> str:
    """
    Executes a shell command and returns its output.

    Args:
        command: The shell command to execute.

    Returns:
        The output of the command as a string, or an error message if the command fails.
    """
    try:
        # Using os.popen to execute the command and capture its stdout.
        # The 'r' mode is for reading the output.
        pipe = os.popen(command, 'r')
        output = pipe.read()
        
        # close() waits for the command to terminate and returns its exit status.
        # On Unix-like systems, a status of 0 or None (for older versions/specifics) means success.
        # On Windows, 0 means success. Non-zero indicates an error.
        exit_status = pipe.close()

        # Check if the command executed successfully
        if exit_status is None or exit_status == 0:
            return output.strip()  # Strip trailing newline often present
        else:
            # Command executed but returned a non-zero exit status, indicating an error.
            # os.popen does not easily capture stderr, so we include stdout in the error message.
            return f"Command '{command}' failed with exit status {exit_status}.\nOutput:\n{output.strip()}"
            
    except OSError as e:
        # This exception can occur if the command is not found or os.popen fails for other OS reasons.
        return f"Failed to execute command '{command}'. OSError: {e}"
    except Exception as e:
        # Catch any other unexpected exceptions.
        return f"An unexpected error occurred while executing command '{command}': {e}"

def filter_and_execute_command(command: str, allowed_subcommands: list[str]) -> str:
    """
    Filters a command against a list of allowed subcommands and executes it if valid.

    Args:
        command: The shell command to execute.
        allowed_subcommands: A list of allowed subcommands and arguments.

    Returns:
        The output of the command as a string, or an error message if the command
        is not allowed or fails.
    """
    try:
        # Parse the command into components. shlex.split handles quotes and spaces.
        command_parts = shlex.split(command)
    except ValueError as e:
        return f"Error parsing command '{command}': {e}"

    if not command_parts:
        return "Error: Empty command provided."

    # Check if all parts of the command are in the allowed list
    for part in command_parts:
        if part not in allowed_subcommands:
            return f"Error: Subcommand or argument '{part}' in command '{command}' is not allowed."

    # If all parts are allowed, execute the command
    return execute_command(command)

def run_shell_command_with_filters(command: str, allowed_subcommands_list: list[str]) -> str:
    """
    Executes a shell command if all its subcommands are in the allowed list,
    handling errors and raising ValueError for invalid or disallowed commands.

    Args:
        command: The shell command to execute.
        allowed_subcommands_list: A list of allowed subcommands and arguments.

    Returns:
        The output of the command as a string.

    Raises:
        ValueError: If the command is empty, cannot be parsed, or contains
                    subcommands/arguments not in the allowed_subcommands_list.
    """
    if not command.strip():
        raise ValueError("Error: Empty command provided.")

    try:
        # Parse the command into components. shlex.split handles quotes and spaces.
        command_parts = shlex.split(command)
    except ValueError as e:
        # Raise a new ValueError to provide a consistent error type from this function
        raise ValueError(f"Error parsing command '{command}': {e}")

    if not command_parts:
        # This case might be redundant due to the initial strip check,
        # but shlex.split on "   " results in [], so it's a good safeguard.
        raise ValueError("Error: Parsed command is empty.")

    # Check if all parts of the command are in the allowed list
    for part in command_parts:
        if part not in allowed_subcommands_list:
            raise ValueError(f"Error: Subcommand or argument '{part}' in command '{command}' is not allowed.")

    # If all parts are allowed, execute the command
    # execute_command already handles its own exceptions and returns error strings
    # or successful output.
    result = execute_command(command)
    
    # Check if execute_command returned an error message (e.g. non-zero exit status)
    # This part depends on how `execute_command` signals errors.
    # Assuming `execute_command` returns a string that starts with "Command '" and "failed with exit status"
    # or "Failed to execute command" or "An unexpected error occurred" for errors.
    if result.startswith("Command '") and "failed with exit status" in result or \
       result.startswith("Failed to execute command") or \
       result.startswith("An unexpected error occurred"):
        # Optionally, one might want to raise a custom exception here too,
        # or just return the error string as per current execute_command behavior.
        # For now, returning the error string from execute_command.
        # If the requirement was to also raise ValueError for execution failures,
        # this part would need to change.
        pass # Let the error string from execute_command pass through

    return result
