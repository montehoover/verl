import subprocess
import shlex

# Provided setup code
base_command = "echo Hello, {user_name}!"

def generate_greeting_command(user_name: str) -> str:
    """
    Generate a shell command that greets the given user.

    Args:
        user_name: The user-provided name to embed into the command.

    Returns:
        The generated shell command as a string.

    Note:
        This function returns a shell command string. It does not execute it.
        The user_name is shell-escaped to reduce the risk of command injection.
    """
    # Ensure we operate on a string and shell-escape it
    safe_user_name = shlex.quote(str(user_name))
    return base_command.format(user_name=safe_user_name)
