import os
import shlex
import subprocess

def filter_and_run_command(cmd_input: str, permitted_subcommands: list) -> str:
    """
    Execute a shell command only if it passes through specific filters.
    
    Args:
        cmd_input: str, the command to execute
        permitted_subcommands: list, a list of subcommands that are considered safe and permitted
    
    Returns:
        str, the output of the successful command or an error message
    
    Raises:
        ValueError: Raised when the command or any of its subcommands do not match the allowed list
    """
    # Parse the command into parts
    try:
        cmd_parts = shlex.split(cmd_input)
    except ValueError as e:
        raise ValueError(f"Invalid command syntax: {e}")
    
    if not cmd_parts:
        raise ValueError("Empty command provided")
    
    # Check if all command parts are in the permitted list
    for part in cmd_parts:
        if part not in permitted_subcommands:
            raise ValueError(f"Command part '{part}' is not in the list of permitted subcommands")
    
    # Execute the command
    try:
        result = subprocess.run(
            cmd_parts,
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"Command failed with error: {e.stderr}"
    except Exception as e:
        return f"Command execution failed: {str(e)}"
