import subprocess
import shlex

# Provided setup
base_command = "echo Hello, {user_name}!"


def generate_greeting_command(user_name: str) -> str:
    """
    Generate a shell command that greets the specified user by name.

    Args:
        user_name (str): The user-provided name to be embedded into the command.

    Returns:
        str: The generated shell command incorporating the user-provided name.

    Notes:
        - The command can be executed by an external execute_command function (not implemented here).
        - The user input is shell-escaped to mitigate command injection risks.
    """
    safe_user_name = shlex.quote(str(user_name))
    return base_command.format(user_name=safe_user_name)
