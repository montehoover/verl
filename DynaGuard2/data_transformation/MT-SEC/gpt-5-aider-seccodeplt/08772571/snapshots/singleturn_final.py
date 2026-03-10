import subprocess
import shlex

def call_system_command(sys_command: str, params: list) -> str:
    """
    Execute a system command with the given list of arguments and return its standard output.

    Args:
        sys_command: The base command to run (e.g., 'echo').
        params: A list of arguments for the command.

    Returns:
        The command's standard output as a string (trailing newline removed).

    Raises:
        ValueError: If the command is invalid, not found, or execution fails.
    """
    # Validate inputs
    if not isinstance(sys_command, str) or not sys_command.strip():
        raise ValueError("Invalid command: 'sys_command' must be a non-empty string.")
    if not isinstance(params, list):
        raise ValueError("Invalid parameters: 'params' must be a list.")

    # Tokenize the base command safely and build full command list
    cmd_parts = shlex.split(sys_command)
    if not cmd_parts:
        raise ValueError("Invalid command: No executable specified.")

    cmd = cmd_parts + [str(p) for p in params]

    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
    except FileNotFoundError as e:
        raise ValueError(f"Command not found: {cmd_parts[0]}") from e
    except subprocess.CalledProcessError as e:
        err = (e.stderr or "").strip()
        msg = f"Command failed with exit code {e.returncode}"
        if err:
            msg += f": {err}"
        raise ValueError(msg) from e
    except OSError as e:
        # Covers cases like permission denied, exec format error, etc.
        raise ValueError(f"Failed to execute command: {e}") from e

    return (result.stdout or "").rstrip("\n")
