import subprocess
import shlex

def execute_command(base_command: str, parameters: list) -> str:
    """
    Execute a shell command with the provided base command and parameters.

    Args:
        base_command: The main command to execute (e.g., 'ls' or 'git status').
        parameters: A list of arguments to pass to the command.

    Returns:
        The standard output (stdout) produced by the command as a string.

    Raises:
        ValueError: If inputs are invalid, the command cannot be found, or the command fails.
    """
    # Validate inputs
    if not isinstance(base_command, str) or not base_command.strip():
        raise ValueError("base_command must be a non-empty string")

    if not isinstance(parameters, list):
        raise ValueError("parameters must be a list")

    # Tokenize the base command to support values like "git status"
    try:
        cmd_parts = shlex.split(base_command)
    except ValueError as e:
        # shlex.split can raise ValueError on malformed input
        raise ValueError(f"Invalid base_command: {e}") from e

    # Ensure all parameters are strings (convert if not)
    try:
        param_parts = [str(p) for p in parameters]
    except Exception as e:
        raise ValueError(f"Invalid parameters: {e}") from e

    full_cmd = cmd_parts + param_parts

    try:
        completed = subprocess.run(
            full_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        return completed.stdout
    except FileNotFoundError as e:
        raise ValueError(f"Command not found: {cmd_parts[0]}") from e
    except subprocess.CalledProcessError as e:
        err = e.stderr if e.stderr is not None else e.stdout
        message = err.strip() if err else str(e)
        raise ValueError(f"Command failed with exit code {e.returncode}: {message}") from e
    except Exception as e:
        raise ValueError(f"Failed to execute command: {e}") from e
