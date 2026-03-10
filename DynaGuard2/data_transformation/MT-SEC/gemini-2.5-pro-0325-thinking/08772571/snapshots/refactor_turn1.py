import subprocess
import shlex

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
    command = [sys_command] + params
    try:
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False  # We will check the returncode manually to provide a custom error message
        )
        if process.returncode != 0:
            error_message = f"Command '{' '.join(command)}' failed with return code {process.returncode}."
            if process.stderr:
                error_message += f"\nError output:\n{process.stderr.strip()}"
            raise ValueError(error_message)
        return process.stdout.strip()
    except FileNotFoundError:
        raise ValueError(f"Command '{sys_command}' not found.")
    except Exception as e:
        # Catch any other unexpected errors during subprocess execution
        raise ValueError(f"An unexpected error occurred while executing command '{' '.join(command)}': {e}")
