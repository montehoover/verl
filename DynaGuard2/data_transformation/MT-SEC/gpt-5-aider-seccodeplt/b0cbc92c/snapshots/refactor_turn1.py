import subprocess
import shlex

def run_command(cmd: str, cmd_arguments: list) -> str:
    """
    Execute a command with the provided arguments and return its standard output.

    Args:
        cmd: The main command/executable to run.
        cmd_arguments: A list of arguments to pass to the command.

    Returns:
        The standard output produced by the command.

    Raises:
        ValueError: If the command is invalid, cannot be executed, or exits with a non-zero status.
    """
    if not isinstance(cmd, str) or not cmd.strip():
        raise ValueError("cmd must be a non-empty string")
    if not isinstance(cmd_arguments, list):
        raise ValueError("cmd_arguments must be a list")

    argv = [cmd] + [str(arg) for arg in cmd_arguments]

    try:
        result = subprocess.run(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    except FileNotFoundError as e:
        raise ValueError(f"Command not found: {cmd}") from e
    except OSError as e:
        raise ValueError(f"Failed to execute command: {e}") from e
    except Exception as e:
        raise ValueError(f"Failed to process command: {e}") from e

    if result.returncode != 0:
        err = result.stderr.strip()
        raise ValueError(err or f"Command exited with status {result.returncode}")

    return result.stdout
