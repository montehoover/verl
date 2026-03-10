import os
import shlex
import subprocess

def parse_command(shell_cmd: str) -> list:
    """
    Parse a shell command into tokens.
    
    Args:
        shell_cmd: The command to parse
        
    Returns:
        List of command tokens
        
    Raises:
        ValueError: When the command syntax is invalid
    """
    try:
        tokens = shlex.split(shell_cmd)
    except ValueError as e:
        raise ValueError(f"Invalid command syntax: {e}")
    
    if not tokens:
        raise ValueError("Empty command provided")
    
    return tokens

def validate_tokens(tokens: list, safe_subcommands: list) -> None:
    """
    Validate that all tokens are in the safe subcommands list.
    
    Args:
        tokens: List of command tokens to validate
        safe_subcommands: List of allowed subcommands
        
    Raises:
        ValueError: When any token is not in the safe subcommands list
    """
    for token in tokens:
        if token not in safe_subcommands:
            raise ValueError(f"Command contains disallowed subcommand: '{token}'")

def execute_command(shell_cmd: str) -> str:
    """
    Execute a shell command and return its output.
    
    Args:
        shell_cmd: The command to execute
        
    Returns:
        The output of the command or an error message
    """
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
    tokens = parse_command(shell_cmd)
    
    # Validate all tokens are safe
    validate_tokens(tokens, safe_subcommands)
    
    # Execute the command if validation passes
    return execute_command(shell_cmd)
