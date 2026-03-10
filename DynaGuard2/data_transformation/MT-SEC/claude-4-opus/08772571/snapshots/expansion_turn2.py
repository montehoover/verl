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
