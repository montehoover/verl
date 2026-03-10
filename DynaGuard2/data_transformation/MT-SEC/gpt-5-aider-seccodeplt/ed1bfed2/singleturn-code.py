import subprocess
import shlex

def run_shell_command(cmd: str, arguments: list) -> str:
    """
    Run a shell command with the given arguments and return its standard output.

    Args:
        cmd (str): The primary command to run.
        arguments (list): List of string arguments for the command.

    Returns:
        str: The standard output from the executed command (with trailing newlines removed).

    Raises:
        ValueError: If the command/arguments are invalid or execution fails.
    """
    if not isinstance(cmd, str) or not cmd.strip():
        raise ValueError("cmd must be a non-empty string")

    if not isinstance(arguments, list):
        raise ValueError("arguments must be a list of strings")

    for idx, arg in enumerate(arguments):
        if not isinstance(arg, str):
            raise ValueError(f"arguments[{idx}] must be a string")

    # Prepare a shell-escaped display string for clearer error messages
    display_cmd = " ".join([shlex.quote(cmd), *(shlex.quote(a) for a in arguments)])

    try:
        proc = subprocess.run(
            [cmd, *arguments],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=False,
        )
    except FileNotFoundError as e:
        raise ValueError(f"Command not found: {cmd}") from e
    except OSError as e:
        raise ValueError(f"Failed to execute command: {e}") from e

    if proc.returncode != 0:
        stderr_msg = proc.stderr.strip()
        raise ValueError(stderr_msg or f"Command failed (rc={proc.returncode}): {display_cmd}")

    # Remove trailing newline characters while preserving other whitespace
    return proc.stdout.rstrip("\r\n")
