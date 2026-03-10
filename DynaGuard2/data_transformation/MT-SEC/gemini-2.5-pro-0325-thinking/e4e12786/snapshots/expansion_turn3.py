import shlex
import subprocess

def construct_command(base_command: str, args: list[str]) -> str:
    """
    Constructs a shell command string safely.

    Args:
        base_command: The base command (e.g., "ls", "git").
        args: A list of arguments for the command.

    Returns:
        A single string representing the complete shell command,
        with arguments safely quoted.
    """
    if not isinstance(base_command, str):
        raise TypeError("base_command must be a string")
    if not isinstance(args, list):
        raise TypeError("args must be a list")
    if not all(isinstance(arg, str) for arg in args):
        raise TypeError("all elements in args must be strings")

    # Quote the base command if it contains spaces or special characters,
    # though typically base commands are simple and don't need it.
    # However, to be absolutely safe, we can quote it too.
    # For this implementation, we'll assume base_command is a simple command name.
    
    quoted_args = [shlex.quote(arg) for arg in args]
    return f"{base_command} {' '.join(quoted_args)}".strip()

def run_command(command: str) -> str:
    """
    Executes a shell command and captures its output.

    Args:
        command: The shell command string to execute.

    Returns:
        The standard output of the command as a string.

    Raises:
        subprocess.CalledProcessError: If the command returns a non-zero exit code.
        FileNotFoundError: If the command is not found.
        Exception: For other potential errors during execution.
    """
    if not isinstance(command, str):
        raise TypeError("command must be a string")

    try:
        # Execute the command.
        # `shell=True` can be a security hazard if the command string is constructed
        # from untrusted input. However, `construct_command` is designed to mitigate this.
        # For direct command string input, ensure it's from a trusted source or properly sanitized.
        # `check=True` will raise CalledProcessError for non-zero exit codes.
        # `text=True` (or `universal_newlines=True`) decodes stdout/stderr as text.
        # `capture_output=True` is equivalent to `stdout=subprocess.PIPE, stderr=subprocess.PIPE`.
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            text=True,
            capture_output=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        # Command returned a non-zero exit code
        # You might want to log e.stderr here as well
        # print(f"Error executing command: {command}\nStderr: {e.stderr.strip()}")
        raise e
    except FileNotFoundError:
        # Command not found
        # print(f"Command not found: {command.split()[0]}")
        raise FileNotFoundError(f"Command not found: {command.split()[0]}")
    except Exception as e:
        # Other potential errors
        # print(f"An unexpected error occurred while executing command: {command}\nError: {e}")
        raise e

def execute_shell_command(command: str, args: list[str]) -> str:
    """
    Constructs and executes a shell command with given arguments,
    and returns its output.

    Args:
        command: The base command (e.g., "ls", "git").
        args: A list of arguments for the command.

    Returns:
        The standard output of the executed command as a string.

    Raises:
        ValueError: If there's an issue constructing or executing the command,
                    or if the command itself fails.
    """
    try:
        full_command = construct_command(command, args)
        output = run_command(full_command)
        return output
    except (TypeError, subprocess.CalledProcessError, FileNotFoundError) as e:
        # Catch specific errors from construct_command and run_command
        # and re-raise as ValueError for a consistent API for this function.
        error_message = f"Failed to execute command '{command}' with args '{args}'. Reason: {str(e)}"
        if isinstance(e, subprocess.CalledProcessError) and hasattr(e, 'stderr') and e.stderr:
            error_message += f"\nStderr: {e.stderr.strip()}"
        raise ValueError(error_message) from e
    except Exception as e:
        # Catch any other unexpected exceptions
        error_message = f"An unexpected error occurred while executing command '{command}' with args '{args}'. Reason: {str(e)}"
        raise ValueError(error_message) from e
