import subprocess
import shlex

def execute_shell_command(command: str, args: list) -> str:
    """
    Execute a shell command with user-provided arguments.

    :param command: str - the base command to execute (e.g., "ls" or "git status")
    :param args: list - a list of arguments for the command
    :return: str - the standard output of the executed command
    :raises ValueError: when the command is invalid or cannot be executed
    """
    if not isinstance(command, str) or not command.strip():
        raise ValueError("Command must be a non-empty string.")
    if not isinstance(args, list):
        raise ValueError("Args must be a list.")

    try:
        cmd_parts = shlex.split(command)
    except ValueError as e:
        # shlex.split can raise ValueError on malformed input
        raise ValueError(f"Invalid command: {e}") from e

    arg_parts = [str(a) for a in args]

    full_cmd = cmd_parts + arg_parts

    try:
        result = subprocess.run(
            full_cmd,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout
    except FileNotFoundError as e:
        raise ValueError(f"Invalid command: {e}") from e
    except PermissionError as e:
        raise ValueError(f"Cannot execute command: {e}") from e
    except subprocess.CalledProcessError as e:
        # Command executed but returned non-zero exit status
        err_output = e.stderr if e.stderr is not None else e.stdout
        raise ValueError(f"Command failed with exit code {e.returncode}: {err_output}") from e
    except OSError as e:
        raise ValueError(f"Execution error: {e}") from e
