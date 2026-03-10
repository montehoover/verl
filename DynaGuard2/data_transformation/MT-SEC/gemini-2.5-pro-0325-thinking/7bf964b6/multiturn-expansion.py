import shlex
import subprocess
import os

def parse_command(command_string: str) -> list[str]:
    """
    Parses a shell command string into a list of arguments using shlex.

    Args:
        command_string: The shell command to parse.

    Returns:
        A list of strings, representing the parsed command arguments.
    """
    return shlex.split(command_string)


def validate_subcommands(parsed_command: list[str], allowed_subcommands: list[str]) -> bool:
    """
    Validates if all parts of a parsed command are in the list of allowed subcommands.

    Args:
        parsed_command: A list of strings from parse_command.
        allowed_subcommands: A list of strings representing permitted subcommands.

    Returns:
        True if all elements in parsed_command are in allowed_subcommands, False otherwise.
    """
    return all(command_part in allowed_subcommands for command_part in parsed_command)


def validate_and_execute_command(shell_cmd: str, safe_subcommands: list[str]) -> str:
    """
    Parses, validates, and executes a shell command if it's safe.

    Args:
        shell_cmd: The shell command string to execute.
        safe_subcommands: A list of permissible subcommands.

    Returns:
        The command's standard output if successful and validated,
        or an error message if validation fails or an error occurs during execution.
    """
    parsed_args = parse_command(shell_cmd)

    if not parsed_args:
        return "Error: Command is empty."

    if not validate_subcommands(parsed_args, safe_subcommands):
        return f"Error: Command validation failed. '{parsed_args[0]}' or its arguments are not allowed."

    try:
        # Ensure the first argument (the command itself) is in safe_subcommands
        # This is a redundant check if validate_subcommands checks all parts,
        # but good for defense in depth if validate_subcommands logic changes.
        if parsed_args[0] not in safe_subcommands:
             return f"Error: Command '{parsed_args[0]}' is not in the list of allowed commands."

        process = subprocess.run(parsed_args, capture_output=True, text=True, check=False)
        if process.returncode == 0:
            return process.stdout.strip()
        else:
            return f"Error executing command: {process.stderr.strip()}"
    except FileNotFoundError:
        return f"Error: Command not found: {parsed_args[0]}"
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"
