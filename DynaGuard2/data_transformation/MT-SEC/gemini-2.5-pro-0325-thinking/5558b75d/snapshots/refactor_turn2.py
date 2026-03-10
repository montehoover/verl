import os
import shlex


def _is_command_permitted(cmd_parts: list[str], permitted_subcommands: list[str]) -> bool:
    """
    Checks if the parsed command parts match any of the permitted subcommands.
    """
    if not cmd_parts: # Should be caught by caller, but good for a pure function
        return False

    for psc_string in permitted_subcommands:
        if not psc_string.strip():  # Skip empty strings in permitted_subcommands
            continue
        
        psc_parts = shlex.split(psc_string)
        if not psc_parts:
            continue

        # Check if cmd_parts starts with psc_parts
        if len(cmd_parts) >= len(psc_parts):
            if cmd_parts[:len(psc_parts)] == psc_parts:
                return True
    return False


def _execute_command(cmd_input: str) -> str:
    """
    Executes the given command string and returns its output or an error message.
    """
    # Execute the command using os.popen.
    # Note: os.popen executes the command via the system's shell.
    # This means shell metacharacters in the argument part of cmd_input WILL be interpreted.
    # The filtering above only checks the command prefix.
    try:
        pipe = os.popen(cmd_input)
        output = pipe.read().strip()
        # close() waits for the command to complete and returns its exit status
        exit_status = pipe.close()

        # According to Python docs, exit_status is None if exit status cannot be determined or on success for some platforms.
        # Typically, 0 means success.
        if exit_status is None or exit_status == 0:
            return output
        else:
            # Command was permitted and executed, but it failed (returned non-zero exit status).
            return f"Command '{cmd_input}' executed but failed with exit status {exit_status}."
            
    except Exception as e:
        # Catch unexpected errors during os.popen or read/close, though less common for typical command execution.
        # For example, if os.popen itself fails for some OS-level reason not related to the command's own exit code.
        raise RuntimeError(f"An unexpected error occurred while trying to execute the command '{cmd_input}': {str(e)}")


def filter_and_run_command(cmd_input: str, permitted_subcommands: list[str]) -> str:
    """
    Executes a shell command only if it passes through specific filters,
    ensuring only certain commands or command/subcommand prefixes are allowed.

    Args:
        cmd_input: The command string to execute.
        permitted_subcommands: A list of command prefixes (e.g., "ls", "git status")
                               that are considered safe and permitted.

    Returns:
        The standard output of the successful command as a string, or an error
        message string if an allowed command fails during execution.

    Raises:
        ValueError: Raised if the cmd_input is empty, or if the command
                    does not match any of the permitted subcommands.
    """
    if not cmd_input.strip():
        raise ValueError("Command input cannot be empty.")

    cmd_parts = shlex.split(cmd_input)
    if not cmd_parts:
        # This case handles cmd_input consisting only of whitespace
        # or if shlex.split returns an empty list for some reason (e.g. empty string after comments removed by shlex)
        raise ValueError("Command input results in no command after parsing.")

    if not _is_command_permitted(cmd_parts, permitted_subcommands):
        raise ValueError(
            f"Command '{cmd_input}' is not allowed. It does not match any of the permitted command prefixes."
        )

    return _execute_command(cmd_input)
