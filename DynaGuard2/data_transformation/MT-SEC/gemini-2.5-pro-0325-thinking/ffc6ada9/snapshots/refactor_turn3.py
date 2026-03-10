import subprocess
import logging
from typing import List

# Configure a logger for this module
logger = logging.getLogger(__name__)


def execute_command(base_command: str, parameters: List[str]) -> str:
    """Runs a shell command with parameters provided by the user.

    Args:
        base_command: The main command that needs to be executed.
        parameters: A list of string arguments to pass alongside the command.

    Returns:
        The standard output produced by the executed shell command, stripped
        of leading/trailing whitespace.

    Raises:
        ValueError: Raised if the base_command is empty, if the command
                    execution fails (e.g., command not found, permission issues,
                    non-zero exit code), or if any other subprocess error occurs.
    """
    if not base_command:
        logger.warning("execute_command called with an empty base_command.")
        raise ValueError("Base command cannot be empty.")

    # Ensure all parameters are strings, as subprocess.run expects them.
    str_parameters = [str(p) for p in parameters]
    command = [base_command] + str_parameters
    command_str_for_logging = ' '.join(command) # For easier reading in logs

    logger.info(f"Executing command: {command_str_for_logging}")

    try:
        # The command is passed as a list of arguments to subprocess.run.
        # Using shell=False (the default) and check=True is generally safer
        # and handles arguments with spaces correctly without manual quoting.
        # check=True will raise CalledProcessError for non-zero exit codes.
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,  # Decodes stdout and stderr using system's default encoding.
            check=True  # Raises CalledProcessError if return code is non-zero.
        )
        logger.info(f"Command '{command_str_for_logging}' executed successfully. Output length: {len(process.stdout.strip())}")
        return process.stdout.strip()

    except subprocess.CalledProcessError as e:
        error_message = (
            f"Command '{' '.join(e.cmd)}' failed with return code {e.returncode}."
        )
        if e.stderr:
            error_message += f"\nError output:\n{e.stderr.strip()}"
        logger.error(f"CalledProcessError for command '{command_str_for_logging}': {error_message}", exc_info=True)
        raise ValueError(error_message) from e
    except FileNotFoundError:
        # This specific exception occurs if the base_command executable is not found.
        logger.error(f"FileNotFoundError: Command not found: {base_command}", exc_info=True)
        raise ValueError(f"Command not found: {base_command}") from None
    except PermissionError:
        # This may occur if the script lacks permissions to execute the command.
        logger.error(f"PermissionError: Permission denied to execute command: {base_command}", exc_info=True)
        raise ValueError(f"Permission denied to execute command: {base_command}") from None
    except Exception as e:
        # Catch any other unexpected errors during subprocess execution.
        # Using ' '.join(command) for the error message as 'e.cmd' might not be available
        # for exceptions other than CalledProcessError.
        logger.error(f"Unexpected exception while executing command '{command_str_for_logging}': {str(e)}", exc_info=True)
        raise ValueError(
            f"An error occurred while executing the command: {command_str_for_logging}. "
            f"Error: {str(e)}"
        ) from e
