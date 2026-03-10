def is_allowed_command(cmd: str) -> bool:
    """
    Checks if a given command string is in a predefined list of allowed commands.
    """
    allowed_commands = ["ls", "pwd", "echo", "cat", "grep"]  # Example list
    return cmd in allowed_commands
