import shlex
import subprocess

def construct_command(base_command, arguments):
    """
    Safely construct a shell command string from a base command and arguments.
    
    Args:
        base_command: The base command as a string
        arguments: List of arguments to append to the command
        
    Returns:
        A properly escaped shell command string
    """
    # Start with the base command
    command_parts = [base_command]
    
    # Add each argument, properly quoted for shell safety
    for arg in arguments:
        command_parts.append(shlex.quote(str(arg)))
    
    # Join all parts with spaces
    return ' '.join(command_parts)

def run_command(command):
    """
    Execute a shell command and return its output.
    
    Args:
        command: The shell command string to execute
        
    Returns:
        The command output as a string
        
    Raises:
        subprocess.CalledProcessError: If the command returns non-zero exit status
        Exception: If the command cannot be executed
    """
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        raise subprocess.CalledProcessError(
            e.returncode,
            e.cmd,
            output=e.stdout,
            stderr=e.stderr
        )
    except Exception as e:
        raise Exception(f"Failed to execute command: {str(e)}")

def execute_shell_command(command, args):
    """
    Construct and execute a shell command with the provided arguments.
    
    Args:
        command: The base command as a string
        args: List of arguments to pass to the command
        
    Returns:
        The command output as a string
        
    Raises:
        ValueError: If any issues arise during command execution
    """
    try:
        # Construct the full command
        full_command = construct_command(command, args)
        
        # Execute the command
        output = run_command(full_command)
        
        return output
    except subprocess.CalledProcessError as e:
        raise ValueError(f"Command failed with exit code {e.returncode}: {e.stderr}")
    except Exception as e:
        raise ValueError(f"Failed to execute command: {str(e)}")
