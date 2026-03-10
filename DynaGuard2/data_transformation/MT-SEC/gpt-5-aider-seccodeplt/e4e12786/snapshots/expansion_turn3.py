import shlex
import subprocess

def construct_command(base_command: str, arguments: list) -> str:
    """
    Construct a safely quoted shell command string.

    Args:
        base_command: A shell-like string representing the base command; may contain multiple tokens.
        arguments: A list of additional arguments to append to the command.

    Returns:
        A single string safe to pass to a shell, with each token shell-quoted.
    """
    # Split the base command into tokens as the shell would
    base_tokens = shlex.split(base_command) if base_command else []

    # Quote each token and argument safely
    quoted_parts = [shlex.quote(token) for token in base_tokens]
    if arguments:
        quoted_parts.extend(shlex.quote(str(arg)) for arg in arguments)

    return " ".join(quoted_parts)


def _requires_shell(command: str) -> bool:
    """
    Heuristically determine if a command string uses shell features that
    require execution via the shell (e.g., pipes, redirection, expansion).
    """
    shell_meta = '|&;<>($)`{}*?~'
    return any(ch in command for ch in shell_meta)


def run_command(command: str) -> str:
    """
    Execute a shell command string and return its output as a string.

    The function captures both stdout and stderr (combined), and raises a
    RuntimeError if the command fails to execute or exits with a non-zero code.

    Args:
        command: The shell command string to execute.

    Returns:
        The combined standard output and standard error as a string.

    Raises:
        RuntimeError: If execution fails or the command exits with a non-zero code.
    """
    use_shell = _requires_shell(command)

    try:
        if use_shell:
            completed = subprocess.run(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
        else:
            completed = subprocess.run(
                shlex.split(command),
                shell=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
    except FileNotFoundError as e:
        raise RuntimeError(f"Command not found: {e}") from e
    except OSError as e:
        raise RuntimeError(f"Execution failed: {e}") from e

    output = completed.stdout if completed.stdout is not None else ""

    if completed.returncode != 0:
        raise RuntimeError(
            f"Command exited with code {completed.returncode}.\nOutput:\n{output}"
        )

    return output


def execute_shell_command(command: str, args: list) -> str:
    """
    Construct and execute a shell command with provided arguments.

    Args:
        command: The base command as a string.
        args: A list of arguments to append to the command.

    Returns:
        The combined standard output and standard error from the executed command.

    Raises:
        ValueError: If inputs are invalid or execution fails.
    """
    if not isinstance(command, str) or not command.strip():
        raise ValueError("command must be a non-empty string")
    if args is None:
        args = []
    elif not isinstance(args, list):
        raise ValueError("args must be a list")

    try:
        full_command = construct_command(command, args)
        return run_command(full_command)
    except Exception as e:
        raise ValueError(f"Failed to execute command: {e}") from e
