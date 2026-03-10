import subprocess
import shlex
import logging


# Configure logger for this module
logger = logging.getLogger(__name__)


def run_command(cmd: str, cmd_arguments: list) -> str:
    """
    Execute a shell command with arguments and return the output.
    
    This function safely executes a shell command with the provided arguments
    by properly handling command injection risks and capturing the output.
    The command and its arguments are processed separately to ensure safe
    execution.
    
    Args:
        cmd (str): The main command to execute. This should be the name
            or path of the executable program to run.
        cmd_arguments (list): A list of arguments to pass to the command.
            Each argument should be a separate element in the list.
            Arguments will be automatically converted to strings if needed.
            
    Returns:
        str: The standard output produced by the executed command.
            If the command produces no output, an empty string is returned.
            
    Raises:
        ValueError: Raised in the following cases:
            - When the command execution fails (non-zero exit code)
            - When the specified command cannot be found in the system
            - When any other error occurs during command processing
            
    Examples:
        >>> run_command("echo", ["Hello", "World"])
        'Hello World\\n'
        
        >>> run_command("ls", ["-la", "/tmp"])
        'total 16\\ndrwxrwxrwt  4 root root 4096 Oct 20 10:30 .\\n...'
        
        >>> run_command("nonexistent", [])
        Traceback (most recent call last):
            ...
        ValueError: Command 'nonexistent' not found
    """
    # Build the full command list with the base command and arguments
    full_command = [cmd] + cmd_arguments
    
    # Log the command being executed
    logger.info(f"Executing command: {cmd} with arguments: {cmd_arguments}")
    logger.debug(f"Full command: {full_command}")
    
    try:
        # Execute the command and capture output
        result = subprocess.run(
            full_command,
            capture_output=True,
            text=True,
            check=True
        )
        
        # Log successful execution
        logger.info(f"Command executed successfully: {cmd}")
        logger.debug(f"Command output (stdout): {result.stdout[:200]}{'...' if len(result.stdout) > 200 else ''}")
        if result.stderr:
            logger.warning(f"Command stderr output: {result.stderr}")
        
        return result.stdout
        
    except subprocess.CalledProcessError as e:
        # Handle non-zero exit codes
        error_message = f"Command execution failed: {e.stderr}"
        logger.error(f"Command failed with exit code {e.returncode}: {cmd}")
        logger.error(f"Error output: {e.stderr}")
        logger.debug(f"Failed command stdout: {e.stdout}")
        raise ValueError(error_message)
        
    except FileNotFoundError:
        # Handle missing commands
        error_message = f"Command '{cmd}' not found"
        logger.error(f"Command not found in system PATH: {cmd}")
        raise ValueError(error_message)
        
    except Exception as e:
        # Handle any other unexpected errors
        error_message = f"Error processing command: {str(e)}"
        logger.exception(f"Unexpected error executing command: {cmd}")
        raise ValueError(error_message)
