import subprocess
import shlex
import os

from typing import List

def execute_shell_command(command: str, args: List[str] = None) -> str:
    """
    Executes a shell command with optional arguments and returns its output.

    Args:
        command: The base command to execute (e.g., "ls", "echo").
        args: A list of arguments for the command (e.g., ["-l", "/tmp"]).
              Defaults to None, meaning no arguments.

    Returns:
        The standard output of the command as a string.

    Raises:
        ValueError: If the command is invalid, fails to execute, or is not found.
    """
    if args is None:
        args = []

    # Ensure all arguments are strings, shlex.quote for safety if any part was to be shell-interpreted
    # However, with shell=False, direct list passing is safer.
    # We will construct the command list directly.
    command_list = [command] + args

    try:
        process = subprocess.run(
            command_list,
            check=True,    # Raises CalledProcessError if return code is non-zero
            capture_output=True,
            text=True,
            shell=False    # Safer: command and args are passed as a list
        )
        return process.stdout.strip()
    except subprocess.CalledProcessError as e:
        error_message = f"Command '{' '.join(command_list)}' failed with exit code {e.returncode}."
        if e.stderr:
            error_message += f" Error: {e.stderr.strip()}"
        else:
            error_message += " No standard error output."
        raise ValueError(error_message)
    except FileNotFoundError:
        # This occurs if the command itself is not found in PATH when shell=False
        raise ValueError(f"Command '{command}' not found. Ensure it is in your PATH or provide an absolute path.")
    except Exception as e: # Catch any other unexpected errors
        raise ValueError(f"An unexpected error occurred while trying to execute '{' '.join(command_list)}': {e}")
