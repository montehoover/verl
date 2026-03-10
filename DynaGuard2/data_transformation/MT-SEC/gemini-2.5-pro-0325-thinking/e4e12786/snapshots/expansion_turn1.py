import shlex

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
