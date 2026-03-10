import subprocess
import shlex

def run_basic_command(command):
    """
    Runs a shell command without arguments and returns its output as a string.
    
    Args:
        command: The shell command to execute
        
    Returns:
        The command output as a string
        
    Raises:
        subprocess.CalledProcessError: If the command execution fails
    """
    result = subprocess.run(
        command,
        shell=True,
        capture_output=True,
        text=True,
        check=True
    )
    return result.stdout

def run_command_with_args(command, args):
    """
    Runs a shell command with arguments and returns its output as a string.
    
    Args:
        command: The shell command to execute
        args: List of arguments to pass to the command
        
    Returns:
        The command output as a string
        
    Raises:
        subprocess.CalledProcessError: If the command execution fails
    """
    # Build the full command with properly escaped arguments
    cmd_parts = [command] + [shlex.quote(str(arg)) for arg in args]
    full_command = ' '.join(cmd_parts)
    
    result = subprocess.run(
        full_command,
        shell=True,
        capture_output=True,
        text=True,
        check=True
    )
    return result.stdout

def call_system_command(sys_command, params):
    """
    Executes a system command with arbitrary arguments dynamically.
    
    Args:
        sys_command: The primary command to execute
        params: List of arguments to pass to the command
        
    Returns:
        The command output as a string
        
    Raises:
        ValueError: If the command is invalid or fails
    """
    try:
        # Validate the command
        if not sys_command or not isinstance(sys_command, str):
            raise ValueError("Invalid command: command must be a non-empty string")
        
        # Build the full command with properly escaped arguments
        cmd_parts = [sys_command] + [shlex.quote(str(param)) for param in params]
        full_command = ' '.join(cmd_parts)
        
        result = subprocess.run(
            full_command,
            shell=True,
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        raise ValueError(f"Command failed with exit code {e.returncode}: {e.stderr}")
    except Exception as e:
        raise ValueError(f"Command execution failed: {str(e)}")
