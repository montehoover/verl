import subprocess
import shlex

def call_system_command(sys_command: str, params: list) -> str:
    """
    Executes a user-specified command with a list of arguments and returns the output as a string.
    Args:
        sys_command (str): The primary command to run (e.g., 'ls').
        params (list): A list of argument strings (e.g., ['-l', '-a']).
    Raises:
        ValueError: If inputs are invalid or the command execution is unsuccessful.
    """
    if not isinstance(sys_command, str) or not sys_command.strip():
        raise ValueError("sys_command must be a non-empty string")

    try:
        cmd_tokens = shlex.split(sys_command)
    except ValueError as e:
        raise ValueError(f"Invalid sys_command: {e}") from e

    if params is None:
        args = []
    else:
        try:
            args = list(params)
        except TypeError as e:
            raise ValueError("params must be an iterable of strings") from e
        if not all(isinstance(a, str) for a in args):
            raise ValueError("All params must be strings")

    try:
        result = subprocess.run(
            [*cmd_tokens, *args],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        err = getattr(e, "stderr", None)
        msg = err if err else str(e)
        raise ValueError(f"Command execution failed: {msg}") from e
