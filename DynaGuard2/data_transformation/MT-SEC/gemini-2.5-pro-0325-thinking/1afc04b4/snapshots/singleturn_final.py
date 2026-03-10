import os
import shlex

def run_shell_command_with_filters(command: str, allowed_subcommands_list: list) -> str:
    """
    Runs a shell command, ensuring all its parts are in an allowed list.

    Args:
        command: The shell command string to execute.
        allowed_subcommands_list: A list of strings. Every token in the parsed
                                  command must be present in this list.

    Returns:
        The standard output of the executed command as a string.

    Raises:
        ValueError: If the command string is invalid (e.g., empty, malformed,
                    or contains parts not in allowed_subcommands_list).
    """
    if not isinstance(command, str):
        # While the prompt focuses on ValueError, basic type safety is good.
        # However, strictly adhering to "Raise ValueError", this could be omitted
        # if inputs are guaranteed to be of the correct type.
        # For robustness, it's included here but could be debated based on context.
        raise TypeError("Command must be a string.")

    if not isinstance(allowed_subcommands_list, list):
        raise TypeError("allowed_subcommands_list must be a list.")
    
    if not all(isinstance(item, str) for item in allowed_subcommands_list):
        raise TypeError("All elements in allowed_subcommands_list must be strings.")

    if not command.strip():  # Handles empty string or string with only whitespace
        raise ValueError("Command cannot be empty.")

    try:
        # shlex.split is robust for parsing command-line arguments
        parsed_command_parts = shlex.split(command)
    except ValueError as e:
        # This handles errors from shlex.split itself, e.g., unmatched quotes
        raise ValueError(f"Invalid command syntax: {e}")

    # If command.strip() is not empty, parsed_command_parts should not be empty
    # with default shlex behavior (e.g., shlex.split("#") -> ['#']).
    # An explicit check for `if not parsed_command_parts:` after shlex.split
    # is generally redundant if the command string itself is confirmed non-empty.
    if not parsed_command_parts and command.strip():
        # This case would be unusual, e.g. if shlex was configured to strip all content.
        # For default shlex, if command.strip() is non-empty, parsed_command_parts will be non-empty.
        raise ValueError("Command parsed to an empty list of components despite non-empty input.")


    for part in parsed_command_parts:
        if part not in allowed_subcommands_list:
            # Provide a clear error message indicating the disallowed part and the original command.
            raise ValueError(f"Disallowed command part: '{part}'. Full command: '{command}'")

    # If all validation passes, execute the command.
    # The 'with' statement ensures the pipe is properly closed after use.
    with os.popen(command) as pipe:
        output = pipe.read()
    
    return output
