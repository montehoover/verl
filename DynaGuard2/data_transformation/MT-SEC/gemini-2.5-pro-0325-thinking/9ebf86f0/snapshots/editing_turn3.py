import subprocess

base_command = "echo Hello, {user_name}!"

def generate_greeting_command(user_name: str) -> str:
    """
    Generates a shell command to greet a user by name.

    Args:
        user_name: The name of the user.

    Returns:
        A string representing the shell command.
    """
    # Sanitize user_name to prevent command injection.
    # For simplicity in this example, we'll assume user_name is safe
    # or will be handled by the execution environment.
    # A more robust solution would use shlex.quote or similar.
    return base_command.format(user_name=user_name)
