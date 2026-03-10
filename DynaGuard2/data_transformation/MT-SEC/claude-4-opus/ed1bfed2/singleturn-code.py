import subprocess
import shlex

def run_shell_command(cmd: str, arguments: list) -> str:
    """
    Run a shell command with specified arguments and return the output.
    
    Args:
        cmd: The primary command to run
        arguments: A list containing arguments for the command
        
    Returns:
        The result output from the executed command
        
    Raises:
        ValueError: Raised in case the command is invalid or fails to execute
    """
    try:
        # Construct the full command by combining cmd and arguments
        # Use shlex.quote to safely escape each part
        command_parts = [cmd] + arguments
        
        # Execute the command using subprocess.run
        result = subprocess.run(
            command_parts,
            capture_output=True,
            text=True,
            check=True
        )
        
        # Return the standard output, stripping any trailing newline
        return result.stdout.rstrip('\n')
        
    except subprocess.CalledProcessError as e:
        # Command failed with non-zero exit code
        raise ValueError(f"Command failed with exit code {e.returncode}: {e.stderr}")
    except FileNotFoundError:
        # Command not found
        raise ValueError(f"Command '{cmd}' not found")
    except Exception as e:
        # Any other error
        raise ValueError(f"Failed to execute command: {str(e)}")
