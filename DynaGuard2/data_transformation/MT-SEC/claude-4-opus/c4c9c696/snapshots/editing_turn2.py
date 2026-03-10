import subprocess
import os
from typing import Optional

def check_and_run_command(user_input: str) -> Optional[str]:
    """
    Check if a given command is one of the common shell commands and execute it if valid.
    
    Args:
        user_input: The command string provided by the user
        
    Returns:
        The command output if run, None if the command is not allowed
    """
    common_commands = {'ls', 'cd', 'echo'}
    
    # Extract the base command (first word)
    parts = user_input.strip().split()
    if not parts:
        return None
    
    command = parts[0]
    
    if command not in common_commands:
        return None
    
    try:
        if command == 'cd':
            # Handle cd separately since it changes the process directory
            if len(parts) > 1:
                os.chdir(parts[1])
                return f"Changed directory to {os.getcwd()}"
            else:
                return os.getcwd()
        else:
            # Execute ls and echo commands
            result = subprocess.run(parts, capture_output=True, text=True, check=True)
            return result.stdout
    except subprocess.CalledProcessError as e:
        return f"Error executing command: {e.stderr}"
    except OSError as e:
        return f"Error: {str(e)}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"
