# Example functions that can be called by commands
def _example_add(a_str: str, b_str: str) -> str:
    """Adds two numbers provided as strings."""
    try:
        result = int(a_str) + int(b_str)
        return str(result)
    except ValueError:
        return "Error: Invalid numbers provided for 'add' operation."

def _example_greet(name: str) -> str:
    """Generates a greeting message."""
    return f"Hello, {name}!"

# Dictionary mapping command strings to callable functions
_AVAILABLE_COMMANDS = {
    "add": _example_add,
    "greet": _example_greet,
}

def validate_command(command_string: str, permitted_subcommands: list[str]) -> str:
    """
    Checks if a given command string contains only allowed subcommands
    and executes it if valid. The first part of the command_string is
    treated as the command name, and subsequent parts as arguments.

    Args:
        command_string: The command string to validate and execute (e.g., "add 1 2").
        permitted_subcommands: A list of all permitted strings that can appear
                               in the command_string (including command names
                               and string representations of arguments).

    Returns:
        The output of the command execution as a string if valid and successful,
        or an error/status message string otherwise.
    """
    if not command_string.strip():
        return "Command not permitted: Empty command."

    subcommand_parts = command_string.split()

    # Validate all parts of the command string
    for part in subcommand_parts:
        if part not in permitted_subcommands:
            return f"Command not permitted: '{part}' is not an allowed subcommand."

    # Proceed to execution if validation passed
    command_name = subcommand_parts[0]
    args = subcommand_parts[1:]

    if command_name in _AVAILABLE_COMMANDS:
        func_to_call = _AVAILABLE_COMMANDS[command_name]
        try:
            # Attempt to call the function with the provided arguments
            result = func_to_call(*args)
            return str(result)  # Ensure the output is a string
        except TypeError:
            # Handles cases like wrong number of arguments
            # For more specific error messages (e.g. expected 2, got 1),
            # 'inspect' module would be needed, which is an import.
            return (f"Error executing command '{command_name}': "
                    f"Incorrect number or type of arguments provided.")
        except Exception as e:
            # Catch any other errors from the command function itself
            return f"Error during execution of command '{command_name}': {e}"
    else:
        # This case means the command_name was in permitted_subcommands
        # but not mapped in _AVAILABLE_COMMANDS.
        return (f"Command '{command_name}' is permitted but not configured "
                f"to be executable.")
