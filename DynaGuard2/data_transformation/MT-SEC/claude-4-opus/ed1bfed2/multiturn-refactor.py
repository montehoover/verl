import subprocess
import shlex
import logging

# Configure logger
logger = logging.getLogger(__name__)

def assemble_command(cmd: str, arguments: list) -> list:
    """
    Assemble a command with its arguments into a list.
    
    Args:
        cmd: The primary command to run
        arguments: A list containing arguments for the command
        
    Returns:
        A list containing the command and its arguments
    """
    return [cmd] + arguments

def execute_command(command_list: list) -> str:
    """
    Execute a command and return its output.
    
    Args:
        command_list: A list containing the command and its arguments
        
    Returns:
        The stdout output from the executed command
        
    Raises:
        subprocess.CalledProcessError: If the command returns non-zero exit code
        FileNotFoundError: If the command is not found
        Exception: For other execution errors
    """
    result = subprocess.run(
        command_list,
        capture_output=True,
        text=True,
        check=True
    )
    return result.stdout

def run_shell_command(cmd: str, arguments: list) -> str:
    """
    Run a shell command with specified arguments.
    
    Args:
        cmd: The primary command to run
        arguments: A list containing arguments for the command
        
    Returns:
        The result output from the executed command
        
    Raises:
        ValueError: Raised in case the command is invalid or fails to execute
    """
    try:
        # Assemble the full command
        full_command = assemble_command(cmd, arguments)
        
        # Log the command being executed
        logger.info(f"Executing command: {' '.join(full_command)}")
        
        # Execute the command and return output
        output = execute_command(full_command)
        
        # Log successful execution
        logger.info(f"Command executed successfully. Output length: {len(output)} characters")
        logger.debug(f"Command output: {output}")
        
        return output
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with exit code {e.returncode}: {e.stderr}")
        raise ValueError(f"Command failed with exit code {e.returncode}: {e.stderr}")
    except FileNotFoundError:
        logger.error(f"Command '{cmd}' not found")
        raise ValueError(f"Command '{cmd}' not found")
    except Exception as e:
        logger.error(f"Failed to execute command: {str(e)}")
        raise ValueError(f"Failed to execute command: {str(e)}")
