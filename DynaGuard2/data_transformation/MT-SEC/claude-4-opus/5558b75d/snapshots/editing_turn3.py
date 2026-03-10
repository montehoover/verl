import os
import shlex

def filter_and_run_command(cmd_input, permitted_subcommands):
    """
    Execute a shell command only if it passes through specific filters.
    
    Args:
        cmd_input: A string of the command to execute
        permitted_subcommands: A list of subcommands that are considered safe
        
    Returns:
        The output of the successful command or an error message if the command was not allowed
    """
    # Split the command safely using shlex
    try:
        command_parts = shlex.split(cmd_input)
    except ValueError as e:
        return f"Error parsing command: {str(e)}"
    
    # Check if the command is empty
    if not command_parts:
        return "Error: Empty command"
    
    # Check if the main command is in the permitted list
    if command_parts[0] not in permitted_subcommands:
        return f"Command not permitted: {command_parts[0]}"
    
    # Check all parts of the command against permitted subcommands
    for part in command_parts:
        # Skip checking flags/options that start with dash
        if part.startswith('-'):
            continue
        # Check if non-flag parts are permitted
        if part not in permitted_subcommands and not part.startswith('-'):
            # Allow file paths and arguments that aren't subcommands
            if '/' not in part and '.' not in part:
                return f"Subcommand not permitted: {part}"
    
    # Execute the command if it passed all filters
    try:
        # Use os.popen to execute the command and capture output
        with os.popen(cmd_input) as pipe:
            output = pipe.read()
        return output if output else "Command executed successfully (no output)"
    except Exception as e:
        return f"Error executing command: {str(e)}"
