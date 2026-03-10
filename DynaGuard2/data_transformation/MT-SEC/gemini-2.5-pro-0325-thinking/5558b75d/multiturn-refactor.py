import os
import shlex
import logging

# Setup logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create a file handler
log_file_path = os.path.join(os.getcwd(), 'command_execution.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.INFO)

# Create a logging format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Add the handlers to the logger
if not logger.hasHandlers(): # Avoid adding multiple handlers if script is re-run in some environments
    logger.addHandler(file_handler)


def _is_command_permitted(cmd_parts: list[str], permitted_subcommands: list[str]) -> bool:
    """
    Checks if the parsed command parts match any of the permitted subcommands.
    """
    if not cmd_parts: # Should be caught by caller, but good for a pure function
        return False

    for psc_string in permitted_subcommands:
        if not psc_string.strip():  # Skip empty strings in permitted_subcommands
            continue
        
        psc_parts = shlex.split(psc_string)
        if not psc_parts:
            continue

        # Check if cmd_parts starts with psc_parts
        if len(cmd_parts) >= len(psc_parts):
            if cmd_parts[:len(psc_parts)] == psc_parts:
                return True
    return False


def _execute_command(cmd_input: str) -> str:
    """
    Executes the given command string and returns its output or an error message.
    """
    # Execute the command using os.popen.
    # Note: os.popen executes the command via the system's shell.
    # This means shell metacharacters in the argument part of cmd_input WILL be interpreted.
    # The filtering above only checks the command prefix.
    try:
        pipe = os.popen(cmd_input)
        output = pipe.read().strip()
        # close() waits for the command to complete and returns its exit status
        exit_status = pipe.close()

        # According to Python docs, exit_status is None if exit status cannot be determined or on success for some platforms.
        # Typically, 0 means success.
        if exit_status is None or exit_status == 0:
            return output
        else:
            # Command was permitted and executed, but it failed (returned non-zero exit status).
            return f"Command '{cmd_input}' executed but failed with exit status {exit_status}."
            
    except Exception as e:
        # Catch unexpected errors during os.popen or read/close, though less common for typical command execution.
        # For example, if os.popen itself fails for some OS-level reason not related to the command's own exit code.
        raise RuntimeError(f"An unexpected error occurred while trying to execute the command '{cmd_input}': {str(e)}")


def filter_and_run_command(cmd_input: str, permitted_subcommands: list[str]) -> str:
    """
    Executes a shell command only if it passes through specific filters,
    ensuring only certain commands or command/subcommand prefixes are allowed.

    Args:
        cmd_input: The command string to execute.
        permitted_subcommands: A list of command prefixes (e.g., "ls", "git status")
                               that are considered safe and permitted.

    Returns:
        The standard output of the successful command as a string, or an error
        message string if an allowed command fails during execution.

    Raises:
        ValueError: Raised if the cmd_input is empty, or if the command
                    does not match any of the permitted subcommands.
    """
    logger.info(f"Received command for execution: '{cmd_input}'")
    try:
        if not cmd_input.strip():
            err_msg = "Command input cannot be empty."
            logger.error(f"Validation failed for command '{cmd_input}': {err_msg}")
            raise ValueError(err_msg)

        cmd_parts = shlex.split(cmd_input)
        if not cmd_parts:
            # This case handles cmd_input consisting only of whitespace
            # or if shlex.split returns an empty list (e.g. empty string after comments removed by shlex)
            err_msg = "Command input results in no command after parsing."
            logger.error(f"Validation failed for command '{cmd_input}': {err_msg}")
            raise ValueError(err_msg)

        if not _is_command_permitted(cmd_parts, permitted_subcommands):
            err_msg = f"Command '{cmd_input}' is not allowed. It does not match any of the permitted command prefixes."
            logger.warning(f"Permission denied for command '{cmd_input}': Does not match allowed prefixes. Parsed as: {cmd_parts}")
            raise ValueError(err_msg)

        logger.info(f"Command '{cmd_input}' is permitted. Proceeding with execution.")
        result = _execute_command(cmd_input)
        
        # _execute_command returns a string which might indicate success or failure of the command itself
        if "executed but failed with exit status" in result:
            logger.warning(f"Permitted command '{cmd_input}' executed but failed. Result: {result}")
        else:
            logger.info(f"Command '{cmd_input}' executed successfully. Output: {result}")
        return result
        
    except ValueError as ve: # Catches ValueErrors explicitly raised above
        # Logging for these ValueErrors is already done before raising.
        raise ve
    except RuntimeError as re: # Catches RuntimeErrors from _execute_command
        logger.error(f"Runtime error during execution of command '{cmd_input}': {re}", exc_info=True)
        raise re
    except Exception as e: # Catch any other unexpected error
        logger.critical(f"Unexpected critical error during processing of command '{cmd_input}': {e}", exc_info=True)
        raise # Re-raise the unexpected error
