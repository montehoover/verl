import subprocess
import shlex
import logging


# Configure logger for this module
logger = logging.getLogger(__name__)


def execute_command(base_command: str, parameters: list) -> str:
    """
    Execute a shell command with parameters and return the output.
    
    This function safely executes a shell command with the provided parameters
    and captures its standard output. It handles proper escaping of arguments
    to prevent shell injection vulnerabilities.
    
    Args:
        base_command (str): The main command that needs to be executed.
            This should be the executable name or path (e.g., 'ls', 'git', '/usr/bin/python').
        parameters (list): A collection of arguments to pass alongside the command.
            Each element in the list will be passed as a separate argument to the command.
            Elements will be automatically converted to strings if they aren't already.
        
    Returns:
        str: The output produced by the executed shell command (stdout).
            If the command produces no output, an empty string is returned.
        
    Raises:
        ValueError: Raised when the command provided is incorrect or cannot be processed.
            This includes cases where:
            - The command executable is not found in the system PATH
            - The command exits with a non-zero status code
            - Any other error occurs during command execution
            
    Examples:
        >>> execute_command('echo', ['Hello', 'World'])
        'Hello World\\n'
        
        >>> execute_command('ls', ['-la', '/tmp'])
        'total 8\\ndrwxrwxrwt  2 root root 4096 Jan  1 00:00 .\\n...'
        
        >>> execute_command('invalid_command', [])
        Traceback (most recent call last):
            ...
        ValueError: Command 'invalid_command' not found
    """
    # Log the command being executed
    logger.info(f"Executing command: '{base_command}' with parameters: {parameters}")
    
    try:
        # Combine command and parameters into a single list
        # No need to quote here as subprocess.run handles it safely when shell=False
        full_command = [base_command] + parameters
        
        logger.debug(f"Full command list: {full_command}")
        
        # Execute the command with subprocess.run
        # capture_output=True captures both stdout and stderr
        # text=True ensures output is returned as string rather than bytes
        # check=True raises CalledProcessError for non-zero exit codes
        result = subprocess.run(
            full_command,
            capture_output=True,
            text=True,
            check=True
        )
        
        # Log successful execution
        logger.info(f"Command executed successfully. Exit code: {result.returncode}")
        
        # Log output details
        if result.stdout:
            logger.debug(f"Command stdout (first 200 chars): {result.stdout[:200]}{'...' if len(result.stdout) > 200 else ''}")
        else:
            logger.debug("Command produced no stdout")
            
        if result.stderr:
            logger.warning(f"Command stderr: {result.stderr}")
        
        # Return the standard output
        return result.stdout
        
    except subprocess.CalledProcessError as e:
        # Handle commands that exit with non-zero status
        logger.error(f"Command failed with exit code {e.returncode}")
        logger.error(f"Command stderr: {e.stderr}")
        logger.debug(f"Command stdout: {e.stdout}")
        raise ValueError(f"Command execution failed: {e}")
    except FileNotFoundError:
        # Handle case where command executable is not found
        logger.error(f"Command executable '{base_command}' not found in PATH")
        raise ValueError(f"Command '{base_command}' not found")
    except Exception as e:
        # Handle any other unexpected errors
        logger.exception(f"Unexpected error while executing command '{base_command}'")
        raise ValueError(f"Error processing command: {e}")
