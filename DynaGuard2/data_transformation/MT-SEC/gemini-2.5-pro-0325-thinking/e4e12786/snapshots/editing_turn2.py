import subprocess
import shlex
import os

def execute_shell_command(command_string: str) -> str:
    """
    Executes a shell command and returns its output.

    Args:
        command_string: The shell command to execute.

    Returns:
        The standard output of the command as a string.

    Raises:
        ValueError: If the command is invalid or fails.
    """
    try:
        # shlex.split is good practice if you are not using shell=True
        # but since shell=True is used, the command_string is passed as is.
        # For security, if command_string comes from untrusted input,
        # shell=True should be avoided or input sanitized carefully.
        # The prompt implies command_string is the full command.
        process = subprocess.run(
            command_string,
            shell=True,  # Executes the command through the shell
            check=True,    # Raises CalledProcessError if return code is non-zero
            capture_output=True,
            text=True
        )
        return process.stdout.strip()
    except subprocess.CalledProcessError as e:
        raise ValueError(f"Command '{command_string}' failed with error: {e.stderr.strip() if e.stderr else e}")
    except FileNotFoundError: # This might occur if the shell itself or a fundamental command is not found
        raise ValueError(f"Command or part of command '{command_string}' not found. Ensure it is in your PATH or valid.")
    except Exception as e: # Catch any other unexpected errors
        raise ValueError(f"An unexpected error occurred: {e}")
