import subprocess
import os
from typing import Optional

def check_and_run_command(user_input: str) -> Optional[str]:
    """
    Checks if a given command is one of the allowed shell commands ('ls', 'cd', 'echo')
    and executes it.

    Args:
        user_input: The command string provided by the user.

    Returns:
        The command output if run, None if the command is not allowed or an error occurs.
    """
    allowed_commands = {'ls', 'echo', 'cd'}
    
    # Split the input string to get the command and its arguments
    command_parts = user_input.strip().split()
    
    if not command_parts:
        return None
        
    command = command_parts[0]
    
    if command not in allowed_commands:
        return None

    try:
        if command == 'cd':
            if len(command_parts) > 1:
                target_dir = command_parts[1]
                os.chdir(target_dir)
                return f"Changed directory to {os.getcwd()}"
            else:
                # 'cd' without arguments typically goes to home,
                # but os.chdir requires a path.
                # For simplicity, let's consider 'cd' without args as an error here
                # or implement going to home directory if desired.
                # For now, returning current working directory.
                return f"Current directory: {os.getcwd()}"
        else:
            # For 'ls' and 'echo', execute using subprocess
            # Using shell=True can be a security risk if user_input is not sanitized.
            # However, since we are strictly controlling the command itself,
            # and only passing arguments, it's somewhat mitigated.
            # For 'echo', arguments are part of the command_parts list.
            # For 'ls', arguments can also be part of command_parts.
            
            # Reconstruct the command with arguments for subprocess
            # Ensure that only the allowed command is executed, arguments are passed safely
            if command == 'echo':
                # Join all parts after 'echo' to form the message
                message_to_echo = " ".join(command_parts[1:])
                # Echo doesn't need shell=True if we handle it this way
                result = subprocess.run([command, message_to_echo], capture_output=True, text=True, check=False)
            else: # for 'ls'
                result = subprocess.run(command_parts, capture_output=True, text=True, check=False)

            if result.returncode == 0:
                return result.stdout.strip()
            else:
                # Return stderr if there was an error
                return f"Error: {result.stderr.strip()}"
                
    except FileNotFoundError:
        return f"Error: Command '{command}' not found."
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"
    
    return None # Should not be reached if logic is correct
