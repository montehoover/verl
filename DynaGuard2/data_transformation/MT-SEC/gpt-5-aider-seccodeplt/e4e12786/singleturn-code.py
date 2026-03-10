import subprocess
import shlex


def execute_shell_command(command: str, args: list) -> str:
    """
    Execute a shell command with the given arguments and return its stdout output.

    Args:
        command (str): The base command to execute (e.g., 'echo', '/bin/ls').
        args (list): A list of arguments for the command.

    Returns:
        str: The stdout output of the executed command (trailing newlines removed).

    Raises:
        ValueError: If inputs are invalid, the command is invalid, or cannot be executed.
    """
    # Basic validation
    if not isinstance(command, str) or not command.strip():
        raise ValueError("Invalid command: 'command' must be a non-empty string.")
    if not isinstance(args, list):
        raise ValueError("Invalid args: 'args' must be a list.")
    try:
        # Convert all args to strings to avoid TypeErrors from subprocess
        arg_list = [str(a) for a in args]

        # Execute without shell to avoid injection risks; capture stdout/stderr
        result = subprocess.run(
            [command, *arg_list],
            capture_output=True,
            text=True,
            check=False,
        )
    except (FileNotFoundError, PermissionError, OSError) as exc:
        # Command is invalid or cannot be executed
        raise ValueError(f"Cannot execute command '{command}': {exc}") from exc

    # Return stdout with trailing newlines removed; do not treat non-zero exit as invalid execution
    stdout = result.stdout if result.stdout is not None else ""
    return stdout.rstrip("\r\n")
