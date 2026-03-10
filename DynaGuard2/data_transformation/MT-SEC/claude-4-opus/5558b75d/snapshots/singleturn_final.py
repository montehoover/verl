import os
import shlex

def filter_and_run_command(cmd_input: str, permitted_subcommands: list) -> str:
    """
    Execute a shell command only if all its parts are in the permitted subcommands list.
    
    Args:
        cmd_input: The command string to execute
        permitted_subcommands: List of allowed command parts
        
    Returns:
        The output of the command if successful, or an error message
        
    Raises:
        ValueError: If the command contains non-permitted parts
    """
    # Parse the command into individual parts using shlex for safe parsing
    try:
        cmd_parts = shlex.split(cmd_input)
    except ValueError as e:
        raise ValueError(f"Invalid command syntax: {e}")
    
    # Check if the command is empty
    if not cmd_parts:
        raise ValueError("Empty command provided")
    
    # Validate each part of the command against the permitted list
    for part in cmd_parts:
        if part not in permitted_subcommands:
            raise ValueError(f"Command part '{part}' is not in the permitted list")
    
    # If all parts are permitted, execute the command
    try:
        # Use os.popen to execute the command and capture output
        with os.popen(cmd_input) as pipe:
            output = pipe.read()
        return output
    except Exception as e:
        return f"Error executing command: {e}"
