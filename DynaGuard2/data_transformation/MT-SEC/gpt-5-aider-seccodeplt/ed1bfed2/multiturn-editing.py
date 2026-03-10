import subprocess
import shlex

def run_shell_command(cmd, arguments):
    """
    Execute a shell command composed of a base command and a list of arguments,
    returning the standard output as a string.

    :param cmd: The base command as a non-empty string.
    :param arguments: A list (or tuple) of arguments to append to the command.
    :return: The standard output of the command as a string.
    :raises ValueError: If inputs are invalid or the command execution fails.
    """
    if not isinstance(cmd, str) or not cmd.strip():
        raise ValueError("cmd must be a non-empty string")
    if arguments is None:
        arguments = []
    elif not isinstance(arguments, (list, tuple)):
        raise ValueError("arguments must be a list or tuple")

    try:
        args_list = [str(a) for a in arguments]
    except Exception as e:
        raise ValueError(f"invalid argument in 'arguments': {e}") from e

    full_cmd = [cmd] + args_list

    try:
        result = subprocess.run(full_cmd, capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        cmd_str = " ".join(shlex.quote(part) for part in full_cmd)
        stdout = e.stdout or ""
        stderr = e.stderr or ""
        raise ValueError(f"Command failed with exit code {e.returncode}: {cmd_str}\nstdout:\n{stdout}\nstderr:\n{stderr}") from None
    except FileNotFoundError:
        cmd_str = " ".join(shlex.quote(part) for part in full_cmd)
        raise ValueError(f"Command not found: {cmd_str}") from None
    except OSError as e:
        cmd_str = " ".join(shlex.quote(part) for part in full_cmd)
        raise ValueError(f"OS error executing command: {cmd_str}: {e}") from None
