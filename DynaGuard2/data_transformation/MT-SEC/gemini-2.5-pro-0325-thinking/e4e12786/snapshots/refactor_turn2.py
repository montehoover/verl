import subprocess
import shlex

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
        raise ValueError("Command must be a string.")
    if not command: # Also check if command string is empty
        raise ValueError("Command cannot be empty.")
    if not isinstance(args, list):
        raise ValueError("Args must be a list of strings.")
    if not all(isinstance(arg, str) for arg in args):
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
    # For error reporting
    command_str_for_reporting = ' '.join(cmd_list)
    # The actual executable part of the command for FileNotFoundError/PermissionError
    original_command_executable = cmd_list[0] if cmd_list else ""

    try:
        process = subprocess.run(
            cmd_list,
            capture_output=True,
            text=True,  # Decodes stdout/stderr to string
            check=False # We will check returncode manually to raise ValueError
        )

        if process.returncode != 0:
            error_message = f"Command '{command_str_for_reporting}' failed with exit code {process.returncode}."
            if process.stderr:
                error_message += f"\nStderr:\n{process.stderr.strip()}"
            raise ValueError(error_message)
        
        return process.stdout.strip()

    except FileNotFoundError:
        raise ValueError(f"Command not found: {original_command_executable}")
    except PermissionError:
        raise ValueError(f"Permission denied to execute command: {original_command_executable}")
    except Exception as e:
        # Catch any other subprocess-related errors or unexpected issues
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
