def run_shell_command(base_command: str, args: list[str]) -> str:
    """
    Assembles a shell command from a base command string and a list of arguments.

    Args:
        base_command: The base command string (e.g., "ls").
        args: A list of arguments (e.g., ["-l", "/home"]).

    Returns:
        The full command as a single string (e.g., "ls -l /home").
    """
    return f"{base_command} {' '.join(args)}"
