import subprocess
import shlex


def _build_command_list(sys_command: str, params: list) -> list:
    """Builds the full command list."""
    return [sys_command] + params


def _execute_command_subprocess(command: list) -> subprocess.CompletedProcess:
    """
    Executes the command using subprocess.run and handles FileNotFoundError.
    """
    try:
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False  # Manual check for more detailed error messages
        )
        return process
    except FileNotFoundError:
        raise ValueError(f"Command '{command[0]}' not found.")
    except Exception as e:
        # Catch any other unexpected errors during subprocess execution
        raise ValueError(
            f"An unexpected error occurred while trying to execute command "
            f"'{' '.join(command)}': {e}"
        )


def _process_command_output(
    process: subprocess.CompletedProcess, command: list
) -> str:
    """
    Processes the subprocess result, raising ValueError on failure.
    Returns stdout on success.
    """
    if process.returncode != 0:
        error_message = (
            f"Command '{' '.join(command)}' failed with "
            f"return code {process.returncode}."
        )
        if process.stderr:
            error_message += f"\nError output:\n{process.stderr.strip()}"
        raise ValueError(error_message)
    return process.stdout.strip()


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
    command_list = _build_command_list(sys_command, params)
    process_result = _execute_command_subprocess(command_list)
    output = _process_command_output(process_result, command_list)
    return output
