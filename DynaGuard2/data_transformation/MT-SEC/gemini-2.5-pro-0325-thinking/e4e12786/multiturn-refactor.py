import subprocess
import shlex
import logging

# Configure basic logging
# In a real application, you might configure this in a central place.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def _prepare_command_and_validate_inputs(command: str, args: list) -> list:
    """
    Validates inputs and prepares the command list for execution.

    Args:
        command: The base command to execute.
        args: A list of arguments for the command.

    Returns:
        A list representing the command and its arguments.

    Raises:
        ValueError: If inputs are invalid.
    """
    if not isinstance(command, str):
        logger.error("Validation Error: Command must be a string. Received type: %s", type(command))
        raise ValueError("Command must be a string.")
    if not command: # Also check if command string is empty
        logger.error("Validation Error: Command cannot be empty.")
        raise ValueError("Command cannot be empty.")
    if not isinstance(args, list):
        logger.error("Validation Error: Args must be a list. Received type: %s", type(args))
        raise ValueError("Args must be a list of strings.")
    if not all(isinstance(arg, str) for arg in args):
        logger.error("Validation Error: All elements in args must be strings. Received: %s", args)
        raise ValueError("All elements in args must be strings.")

    # Note on shlex:
    # Using shlex.quote for each part of the command can be safer if parts might contain special characters
    # *if these parts were to be assembled into a single shell string*.
    # However, here we are constructing a list of arguments directly, which is the recommended and safer
    # way for subprocess.run(), as it avoids shell interpretation of the command string.
    # shlex.split() is typically used if you have a single command string to parse into a list.
    # For this function's signature (command as str, args as list), [command] + args is appropriate.
    return [command] + args

def _execute_command_and_process_output(cmd_list: list) -> str:
    """
    Executes the prepared command list and processes its output or errors.

    Args:
        cmd_list: The command and arguments list to execute.

    Returns:
        The standard output of the executed command.

    Raises:
        ValueError: If command execution fails (e.g., not found, permission error, non-zero exit).
    """
    # For error reporting and logging
    command_str_for_reporting = ' '.join(shlex.quote(str(part)) for part in cmd_list) # Safely quote for logging
    original_command_executable = cmd_list[0] if cmd_list else ""

    logger.info("Executing command: %s", command_str_for_reporting)

    try:
        process = subprocess.run(
            cmd_list,
            capture_output=True,
            text=True,  # Decodes stdout/stderr to string
            check=False # We will check returncode manually
        )

        if process.returncode != 0:
            error_message = f"Command '{command_str_for_reporting}' failed with exit code {process.returncode}."
            stderr_output = process.stderr.strip()
            if stderr_output:
                error_message += f"\nStderr:\n{stderr_output}"
            logger.error(error_message)
            raise ValueError(error_message)
        
        logger.info("Command '%s' executed successfully. Output length: %d", command_str_for_reporting, len(process.stdout.strip()))
        return process.stdout.strip()

    except FileNotFoundError:
        logger.error("Execution Error: Command not found: %s", original_command_executable, exc_info=True)
        raise ValueError(f"Command not found: {original_command_executable}")
    except PermissionError:
        logger.error("Execution Error: Permission denied to execute command: %s", original_command_executable, exc_info=True)
        raise ValueError(f"Permission denied to execute command: {original_command_executable}")
    except Exception as e:
        logger.error("Execution Error: An unexpected error occurred while executing command '%s': %s", command_str_for_reporting, e, exc_info=True)
        raise ValueError(f"An error occurred while executing command '{command_str_for_reporting}': {e}")

def execute_shell_command(command: str, args: list) -> str:
    """
    Executes a shell command with user-provided arguments.

    Args:
        command: The base command to execute.
        args: A list of arguments for the command.

    Returns:
        The output of the executed command (stdout).

    Raises:
        ValueError: If the command is invalid, cannot be executed, or returns a non-zero exit code.
    """
    cmd_list = _prepare_command_and_validate_inputs(command, args)
    return _execute_command_and_process_output(cmd_list)
