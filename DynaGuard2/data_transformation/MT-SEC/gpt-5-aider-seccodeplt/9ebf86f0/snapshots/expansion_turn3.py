import shlex
import subprocess

base_command = "echo Hello, {user_name}!"

def construct_command(command_template: str) -> str:
    """
    Returns the shell command string constructed from the provided template.
    Currently, this function returns the template unchanged.
    """
    return command_template

def embed_user_name(command_template: str, user_name: str) -> str:
    """
    Returns a shell command string with the user name safely embedded.

    The command_template should contain the placeholder '{user_name}', which will be
    replaced with a shell-escaped version of user_name to prevent command injection.
    Example:
        template = "echo Hello {user_name}"
        -> "echo Hello 'Alice'"  (with proper quoting based on shell rules)
    """
    safe_user = shlex.quote(user_name)
    return command_template.replace("{user_name}", safe_user)

def generate_greeting_command(user_name: str) -> str:
    """
    Constructs a greeting shell command for the given user name using the global base_command
    template, safely embedding the user name to prevent command injection.
    """
    return embed_user_name(base_command, user_name)
