import os
import shlex
import subprocess

def validate_and_execute_command(shell_cmd):
    # Parse the command
    try:
        parsed_cmd = shlex.split(shell_cmd)
    except ValueError as e:
        raise ValueError(f"Invalid command syntax: {str(e)}")
    
    # Check if command is empty
    if not parsed_cmd:
        raise ValueError("Command cannot be empty")
    
    # Get the base command
    base_command = parsed_cmd[0]
    
    # Check if it starts with a valid command name pattern
    if not base_command or not base_command[0].isalpha():
        raise ValueError(f"Invalid command name: '{base_command}'")
    
    # Execute the command
    try:
        result = subprocess.run(shell_cmd, shell=True, capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        raise ValueError(f"Command execution failed with return code {e.returncode}: {e.stderr}")
    except Exception as e:
        raise ValueError(f"Command execution failed: {str(e)}")
