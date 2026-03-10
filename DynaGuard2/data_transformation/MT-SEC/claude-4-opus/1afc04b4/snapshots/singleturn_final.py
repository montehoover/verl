import os
import shlex

def run_shell_command_with_filters(command: str, allowed_subcommands_list: list) -> str:
    """
    Run a shell command with pre-applied filters to only allow certain subcommands.
    
    Args:
        command: str - The shell command to execute
        allowed_subcommands_list: list - A list of allowed subcommands that can be run
    
    Returns:
        str - The output of the command or an error message
    
    Raises:
        ValueError - Raised when the command is invalid or contains disallowed subcommands
    """
    # Parse the command into tokens using shlex for safe splitting
    try:
        command_tokens = shlex.split(command)
    except ValueError as e:
        raise ValueError(f"Invalid command syntax: {e}")
    
    # Check if command is empty
    if not command_tokens:
        raise ValueError("Command cannot be empty")
    
    # Validate that all tokens in the command are in the allowed list
    for token in command_tokens:
        if token not in allowed_subcommands_list:
            raise ValueError(f"Disallowed subcommand: '{token}'")
    
    # Execute the command using os.popen
    try:
        # Execute the command and read the output
        with os.popen(command) as pipe:
            output = pipe.read()
        return output
    except Exception as e:
        return f"Error executing command: {e}"
