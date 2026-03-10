import subprocess
import shlex

def run_command(cmd: str, cmd_arguments: list) -> str:
    """
    Run a shell command with arguments and return its standard output as a string.

    Args:
        cmd (str): The main command to execute.
        cmd_arguments (list): A collection of arguments to pass alongside the command.

    Returns:
        str: The output produced by the executed shell command.

    Raises:
        ValueError: If the command is incorrect, cannot be processed, or execution fails.
    """
    if not isinstance(cmd, str) or not cmd.strip():
        raise ValueError("cmd must be a non-empty string.")
    if not isinstance(cmd_arguments, list):
        raise ValueError("cmd_arguments must be a list.")

    try:
        argv = [cmd] + [str(arg) for arg in cmd_arguments]
    except Exception as e:
        raise ValueError("cmd_arguments must be a list of arguments that can be converted to strings.") from e

    try:
        completed = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            check=True,
        )
        # Remove trailing newline(s) to match typical expectations like `echo`.
        return completed.stdout.rstrip("\n")
    except FileNotFoundError as e:
        # Command does not exist on the system.
        raise ValueError(f"Command not found: {cmd}") from e
    except subprocess.CalledProcessError as e:
        # Non-zero exit code; include stderr for clarity.
        cmd_str = " ".join(shlex.quote(part) for part in argv)
        stderr = (e.stderr or "").strip()
        msg = f"Command failed with exit code {e.returncode}: {cmd_str}"
        if stderr:
            msg += f" | stderr: {stderr}"
        raise ValueError(msg) from e
    except Exception as e:
        # Any other unexpected failure.
        cmd_str = " ".join(shlex.quote(part) for part in argv)
        raise ValueError(f"Failed to execute command: {cmd_str}") from e
