import subprocess
import shlex
import logging
import os

# Setup logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create a file handler to log to a file in the current working directory
log_file_path = os.path.join(os.getcwd(), "command_execution.log")
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.INFO)

# Create a logging format
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
file_handler.setFormatter(formatter)

# Add the handlers to the logger
logger.addHandler(file_handler)


def _build_command_list(sys_command: str, params: list) -> list:
    """Builds the full command list."""
    return [sys_command] + params


def _execute_command_subprocess(command: list) -> subprocess.CompletedProcess:
    """
    Executes the command using subprocess.run and handles FileNotFoundError.
    """
    logger.debug(f"Executing subprocess with command list: {command}")
    try:
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False  # Manual check for more detailed error messages
        )
        return process
    except FileNotFoundError:
        error_msg = f"Command '{command[0]}' not found. Full command: {' '.join(command)}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    except Exception as e:
        # Catch any other unexpected errors during subprocess execution
        error_msg = (
            f"An unexpected error occurred while trying to execute command "
            f"'{' '.join(command)}': {e}"
        )
        logger.error(error_msg)
        raise ValueError(error_msg)


def _process_command_output(
    process: subprocess.CompletedProcess, command: list
) -> str:
    """
    Processes the subprocess result, raising ValueError on failure.
    Returns stdout on success.
    """
    if process.returncode != 0:
        error_detail = f"Command '{' '.join(command)}' failed with return code {process.returncode}."
        if process.stderr:
            error_detail += f"\nError output:\n{process.stderr.strip()}"
        logger.error(error_detail)
        raise ValueError(error_detail)

    output = process.stdout.strip()
    logger.debug(
        f"Command '{' '.join(command)}' executed successfully. "
        f"Output:\n{output}"
    )
    return output


def call_system_command(sys_command: str, params: list) -> str:
    """
    Runs a shell command with user-specified arguments.

    Args:
        sys_command: The primary command to run.
        params: A list containing arguments for the command.

    Returns:
        The result output from the executed command.

    Raises:
        ValueError: Raised in case the command is invalid or fails to execute.
    """
    logger.info(
        f"Attempting to run command: '{sys_command}' with params: {params}"
    )
    try:
        command_list = _build_command_list(sys_command, params)
        process_result = _execute_command_subprocess(command_list)
        output = _process_command_output(process_result, command_list)
        logger.info(
            f"Command '{sys_command}' executed successfully. "
            f"Output length: {len(output)} chars."
        )
        # To avoid logging potentially very large outputs to INFO,
        # we log length. Detailed output is at DEBUG in _process_command_output.
        return output
    except ValueError as e:
        logger.error(
            f"ValueError during execution of command '{sys_command}' "
            f"with params {params}: {e}"
        )
        raise
    except Exception as e:
        logger.critical(
            f"Unexpected critical error during execution of command '{sys_command}' "
            f"with params {params}: {e}",
            exc_info=True
        )
        raise ValueError(
            f"An unexpected critical error occurred for command '{sys_command}': {e}"
        )
