def generate_greeting_shell_command(user_name: str) -> str:
    """
    Generates a shell command to print a greeting message for a user.

    Args:
        user_name: The name of the user.

    Returns:
        A string representing the shell command.
    """
    greeting = f"Hello, {user_name}!"
    return f"echo '{greeting}'"
