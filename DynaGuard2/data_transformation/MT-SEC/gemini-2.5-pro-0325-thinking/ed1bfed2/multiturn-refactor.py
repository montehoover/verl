import subprocess
import shlex
import logging

# Configure basic logging
# In a real application, you might configure this in a central place.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def _prepare_command_list(cmd: str, arguments: list) -> list[str]:
    """
    Prepares the command list by combining the command and its arguments,
    ensuring all parts are strings.
    """
    command_list = [cmd] + arguments
    # Ensure arguments are strings, as subprocess.run expects them.
    return [str(arg) for arg in command_list]

def _execute_prepared_command(command_list_str: list[str], original_cmd: str) -> str:
    """
    Executes the prepared command list using subprocess.run and handles exceptions.
    """
    full_command_str = shlex.join(command_list_str)
    logger.info(f"Executing command: {full_command_str}")
    try:
        result = subprocess.run(
            command_list_str,
            capture_output=True,
            text=True,
            check=True  # This will raise CalledProcessError for non-zero exit codes
        )
        logger.info(f"Command '{full_command_str}' executed successfully. Output:\n{result.stdout}")
        return result.stdout
    except subprocess.CalledProcessError as e:
        error_message = f"Command '{full_command_str}' failed with exit code {e.returncode}.\nStderr: {e.stderr}"
        logger.error(error_message)
        raise ValueError(error_message) from e
    except FileNotFoundError:
        error_message = f"Command not found: {original_cmd} (Full attempted command: '{full_command_str}')"
        logger.error(error_message)
        raise ValueError(error_message) from None
    except Exception as e:
        # Catch any other unexpected errors during subprocess execution
        error_message = f"An unexpected error occurred while trying to run command '{full_command_str}': {e}"
        logger.error(error_message, exc_info=True) # exc_info=True will log the stack trace
        raise ValueError(error_message) from e

def run_shell_command(cmd: str, arguments: list) -> str:
    """
    Runs a shell command with user-specified arguments.

    Args:
        cmd: The primary command to run.
        arguments: A list containing arguments for the command.

    Returns:
        The result output (stdout) from the executed command.

    Raises:
        ValueError: Raised in case the command is invalid or fails to execute.
    """
    command_list_str = _prepare_command_list(cmd, arguments)
    return _execute_prepared_command(command_list_str, cmd)
