import os
import shlex

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
    if not command or not command.strip():
        raise ValueError("Command cannot be empty")
    
    # Parse the command safely
    try:
        parsed_command = shlex.split(command)
    except ValueError as e:
        raise ValueError(f"Invalid command syntax: {e}")
    
    if not parsed_command:
        raise ValueError("Command cannot be empty")
    
    # Extract the main command and subcommand
    main_command = parsed_command[0]
    subcommand = parsed_command[1] if len(parsed_command) > 1 else None
    
    # Check if subcommand is allowed
    if subcommand and subcommand not in allowed_subcommands_list:
        raise ValueError(f"Subcommand '{subcommand}' is not allowed. Allowed subcommands: {allowed_subcommands_list}")
    
    # If no subcommand but allowed_subcommands_list is not empty, it means a subcommand is required
    if not subcommand and allowed_subcommands_list:
        raise ValueError(f"A subcommand is required. Allowed subcommands: {allowed_subcommands_list}")
    
    # Execute the command
    try:
        result = os.popen(command).read()
        return result.strip() if result else ""
    except Exception as e:
        return f"Error executing command: {e}"
