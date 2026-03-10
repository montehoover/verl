import subprocess
import shlex

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
    if not isinstance(command, str):
        raise ValueError("Command must be a string.")
    if not isinstance(args, list):
        raise ValueError("Args must be a list of strings.")
    if not all(isinstance(arg, str) for arg in args):
        raise ValueError("All elements in args must be strings.")

    cmd_list = [command] + args

    try:
        # Using shlex.quote for each part of the command can be safer if parts might contain special characters,
        # but here we are constructing a list directly, which is the recommended way for subprocess.
        # shlex.split() is typically used if you have a single command string to parse.
        # For this function signature, [command] + args is appropriate.
        
        process = subprocess.run(
            cmd_list,
            capture_output=True,
            text=True,  # Decodes stdout/stderr to string
            check=False # We will check returncode manually to raise ValueError
        )

        if process.returncode != 0:
            error_message = f"Command '{' '.join(cmd_list)}' failed with exit code {process.returncode}."
            if process.stderr:
                error_message += f"\nStderr:\n{process.stderr.strip()}"
            raise ValueError(error_message)
        
        return process.stdout.strip()

    except FileNotFoundError:
        raise ValueError(f"Command not found: {command}")
    except PermissionError:
        raise ValueError(f"Permission denied to execute command: {command}")
    except Exception as e:
        # Catch any other subprocess-related errors or unexpected issues
        raise ValueError(f"An error occurred while executing command '{' '.join(cmd_list)}': {e}")
