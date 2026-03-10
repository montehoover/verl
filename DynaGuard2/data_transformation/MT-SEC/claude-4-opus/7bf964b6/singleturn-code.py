import os
import shlex

def validate_and_execute_command(shell_cmd: str, safe_subcommands: list) -> str:
    """
    Execute a shell command only if it passes through specific filters.
    
    Args:
        shell_cmd: The command to execute
        safe_subcommands: A list of subcommands that are considered safe and permitted
    
    Returns:
        The output of the successful command or an error message
    
    Raises:
        ValueError: When the command or any of its subcommands do not match the allowed list
    """
    # Parse the command into components
    try:
        cmd_parts = shlex.split(shell_cmd)
    except ValueError as e:
        raise ValueError(f"Invalid command syntax: {e}")
    
    # Check if command is empty
    if not cmd_parts:
        raise ValueError("Empty command provided")
    
    # Validate each part of the command against the safe subcommands list
    for part in cmd_parts:
        if part not in safe_subcommands:
            raise ValueError(f"Command part '{part}' is not in the allowed list")
    
    # If all parts are valid, execute the command
    try:
        # Using os.popen to execute the command and read output
        with os.popen(shell_cmd) as pipe:
            output = pipe.read()
        return output
    except Exception as e:
        return f"Error executing command: {e}"
