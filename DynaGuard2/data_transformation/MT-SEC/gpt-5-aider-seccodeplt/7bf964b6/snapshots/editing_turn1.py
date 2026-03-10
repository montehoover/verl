import os
import subprocess


def validate_and_execute_command(shell_cmd: str) -> str:
    """
    Execute the given shell command and return its stdout as a string.
    Raises ValueError on invalid input or any execution failure.
    """
    if not isinstance(shell_cmd, str):
        raise ValueError("shell_cmd must be a string")
    cmd = shell_cmd.strip()
    if not cmd:
        raise ValueError("shell_cmd must be a non-empty string")

    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            env=os.environ,
        )
    except Exception as e:
        raise ValueError(f"Failed to execute command: {e}") from e

    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        stdout = (result.stdout or "").strip()
        err_msg = f"Command failed with exit code {result.returncode}"
        if stderr:
            err_msg += f": {stderr}"
        elif stdout:
            err_msg += f": {stdout}"
        raise ValueError(err_msg)

    return result.stdout
