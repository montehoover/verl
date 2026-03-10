import subprocess
import shlex

def execute_command(base_command: str, parameters: list) -> str:
    """
    Execute a shell command with provided parameters.

    Args:
        base_command (str): The main command to execute. May include flags.
        parameters (list): Collection of arguments to pass alongside the command.

    Returns:
        str: The standard output produced by the executed command (without trailing newline).

    Raises:
        ValueError: If the command is invalid, fails to execute, or returns a non-zero exit status.
    """
    # Validate inputs
    if not isinstance(base_command, str) or not base_command.strip():
        raise ValueError("base_command must be a non-empty string")

    if not isinstance(parameters, list):
        raise ValueError("parameters must be a list")

    try:
        # Tokenize the base command to respect quoted segments
        cmd_tokens = shlex.split(base_command)
        # Coerce parameters to strings to be safe
        arg_tokens = [str(p) for p in parameters]
        cmd = cmd_tokens + arg_tokens

        completed = subprocess.run(
            cmd,
            text=True,
            capture_output=True,
            check=True
        )
        # Remove trailing newline characters for a cleaner return, matching the example
        return completed.stdout.rstrip("\r\n")
    except (subprocess.CalledProcessError, FileNotFoundError, PermissionError, OSError) as exc:
        # Include stderr if available to aid debugging
        stderr = ""
        if isinstance(exc, subprocess.CalledProcessError):
            stderr = exc.stderr or ""
        elif hasattr(exc, "stderr"):
            stderr = getattr(exc, "stderr") or ""
        msg = f"Failed to execute command: {base_command}"
        if stderr:
            msg += f" | Error: {stderr.strip()}"
        raise ValueError(msg) from exc
