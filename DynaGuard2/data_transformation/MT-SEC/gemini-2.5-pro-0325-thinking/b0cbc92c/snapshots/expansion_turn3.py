import shlex
import subprocess

def construct_command(main_command: str, args: list[str]) -> str:
    """
    Constructs a shell command string from a main command and a list of arguments.

    Args:
        main_command: The main command to execute.
        args: A list of arguments for the command.

    Returns:
        A string representing the full command, with components properly quoted.
    """
    if not isinstance(main_command, str):
        raise TypeError("main_command must be a string")
    if not isinstance(args, list):
        raise TypeError("args must be a list")
    if not all(isinstance(arg, str) for arg in args):
        raise TypeError("All elements in args must be strings")

    quoted_main_command = shlex.quote(main_command)
    quoted_args = [shlex.quote(arg) for arg in args]
    
    return " ".join([quoted_main_command] + quoted_args)


def parse_command(command_string: str) -> list[str]:
    """
    Parses a command string into a list of command components.

    Args:
        command_string: The command string to parse.

    Returns:
        A list of command components.
    """
    if not isinstance(command_string, str):
        raise TypeError("command_string must be a string")
    
    return shlex.split(command_string)


def run_command(cmd: str, cmd_arguments: list[str]) -> str:
    """
    Constructs, parses, and runs a shell command, returning its output.

    Args:
        cmd: The main command to execute.
        cmd_arguments: A list of arguments for the command.

    Returns:
        The standard output of the command as a string.

    Raises:
        TypeError: If cmd is not a string or cmd_arguments is not a list of strings.
        ValueError: If the command is empty after construction, not found,
                    fails to execute due to OS errors (e.g., permissions),
                    or returns a non-zero exit code.
    """
    if not isinstance(cmd, str):
        raise TypeError("cmd must be a string")
    if not isinstance(cmd_arguments, list):
        raise TypeError("cmd_arguments must be a list of strings")
    if not all(isinstance(arg, str) for arg in cmd_arguments):
        raise TypeError("All elements in cmd_arguments must be strings")

    full_command_str = construct_command(cmd, cmd_arguments)
    command_parts = parse_command(full_command_str)

    if not command_parts:
        raise ValueError("Constructed command is empty and cannot be run.")

    try:
        process = subprocess.run(
            command_parts,
            capture_output=True,
            text=True,
            check=False  # We check returncode manually to provide a ValueError
        )

        if process.returncode != 0:
            error_message = (
                f"Command '{' '.join(command_parts)}' failed with "
                f"exit code {process.returncode}."
            )
            if process.stderr:
                error_message += f"\nStderr:\n{process.stderr.strip()}"
            raise ValueError(error_message)
        
        return process.stdout.strip()

    except FileNotFoundError:
        raise ValueError(f"Command not found: '{command_parts[0]}'")
    except PermissionError:
        raise ValueError(f"Permission denied when trying to execute: '{command_parts[0]}'")
    except OSError as e:
        # Catches other OS-level errors during process execution attempt
        raise ValueError(f"OS error executing command '{command_parts[0]}': {e}")
