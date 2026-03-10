import os
import shlex
import subprocess

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
    # Parse the command into tokens
    try:
        tokens = shlex.split(shell_cmd)
    except ValueError as e:
        raise ValueError(f"Invalid command syntax: {e}")
    
    if not tokens:
        raise ValueError("Empty command provided")
    
    # Check if all tokens are in the safe subcommands list
    for token in tokens:
        if token not in safe_subcommands:
            raise ValueError(f"Command contains disallowed subcommand: '{token}'")
    
    # Execute the command if validation passes
    try:
        result = subprocess.run(
            shell_cmd,
            shell=True,
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"Command execution failed: {e.stderr}"
    except Exception as e:
        return f"Error executing command: {str(e)}"
