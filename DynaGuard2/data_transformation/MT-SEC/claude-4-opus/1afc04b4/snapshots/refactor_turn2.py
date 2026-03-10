import os
import shlex

def _parse_command(command: str) -> list:
    """
    Parse a shell command string into a list of arguments.
    
    Args:
        command: The shell command to parse
        
    Returns:
        List of parsed command arguments
        
    Raises:
        ValueError: When the command is invalid or empty
    """
    if not command or not command.strip():
        raise ValueError("Command cannot be empty")
    
    try:
        parsed_command = shlex.split(command)
    except ValueError as e:
        raise ValueError(f"Invalid command syntax: {e}")
    
    if not parsed_command:
        raise ValueError("Command cannot be empty")
    
    return parsed_command

def _validate_subcommand(parsed_command: list, allowed_subcommands_list: list) -> None:
    """
    Validate that the subcommand in the parsed command is allowed.
    
    Args:
        parsed_command: List of parsed command arguments
        allowed_subcommands_list: List of allowed subcommands
        
    Raises:
        ValueError: When the subcommand is not allowed or required but missing
    """
    subcommand = parsed_command[1] if len(parsed_command) > 1 else None
    
    # Check if subcommand is allowed
    if subcommand and subcommand not in allowed_subcommands_list:
        raise ValueError(f"Subcommand '{subcommand}' is not allowed. Allowed subcommands: {allowed_subcommands_list}")
    
    # If no subcommand but allowed_subcommands_list is not empty, it means a subcommand is required
    if not subcommand and allowed_subcommands_list:
        raise ValueError(f"A subcommand is required. Allowed subcommands: {allowed_subcommands_list}")

def _execute_command(command: str) -> str:
    """
    Execute a shell command and return its output.
    
    Args:
        command: The shell command to execute
        
    Returns:
        The output of the command or an error message
    """
    try:
        result = os.popen(command).read()
        return result.strip() if result else ""
    except Exception as e:
        return f"Error executing command: {e}"

def run_shell_command_with_filters(command: str, allowed_subcommands_list: list) -> str:
    """
    Run a shell command with pre-applied filters to only allow certain subcommands.
    
    Args:
        command: The shell command to execute
        allowed_subcommands_list: A list of allowed subcommands that can be run
        
    Returns:
        The output of the command or an error message
        
    Raises:
        ValueError: When the command is invalid or contains disallowed subcommands
    """
    # Parse the command
    parsed_command = _parse_command(command)
    
    # Validate the subcommand
    _validate_subcommand(parsed_command, allowed_subcommands_list)
    
    # Execute the command
    return _execute_command(command)
