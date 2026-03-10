import os
import shlex

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
        raise ValueError("Command input results in no command after parsing.")

    is_permitted = False
    for psc_string in permitted_subcommands:
        if not psc_string.strip():  # Skip empty strings in permitted_subcommands
            continue
        
        psc_parts = shlex.split(psc_string)
        if not psc_parts: # Should ideally not happen if psc_string.strip() is non-empty
            continue

        # Check if cmd_parts starts with psc_parts
        if len(cmd_parts) >= len(psc_parts):
            if cmd_parts[:len(psc_parts)] == psc_parts:
                is_permitted = True
                break
    
    if not is_permitted:
        raise ValueError(
            f"Command '{cmd_input}' is not allowed. It does not match any of the permitted command prefixes."
        )

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
