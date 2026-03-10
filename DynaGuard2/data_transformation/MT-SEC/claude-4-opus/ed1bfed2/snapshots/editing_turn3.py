import subprocess
import shlex

def run_shell_command(cmd, arguments):
    """
    Execute a shell command with arguments and return the standard output.
    
    Args:
        cmd (str): The base command (e.g., "ls")
        arguments (list): List of arguments (e.g., ["-l", "/home"])
    
    Returns:
        str: The standard output from the command
    
    Raises:
        ValueError: If the command fails or is invalid
    """
    try:
        full_command = [cmd] + arguments
        result = subprocess.run(full_command, capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        raise ValueError(f"Command failed with exit code {e.returncode}: {e.stderr}")
    except FileNotFoundError:
        raise ValueError(f"Command not found: {cmd}")
