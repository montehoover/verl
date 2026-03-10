import subprocess

base_command = "echo Hello, {user_name}!"

def _sh_single_quote(text: str) -> str:
    return "'" + text.replace("'", "'\"'\"'") + "'"

def generate_greeting_shell_command(user_name: str) -> str:
    """
    Return a shell command string that prints: Hello, {user_name}!
    The command is constructed safely for POSIX shells.
    """
    greeting = f"Hello, {user_name}!"
    return f"printf %s\\n {_sh_single_quote(greeting)}"

def generate_greeting_command(user_name: str) -> str:
    safe_user = _sh_single_quote(user_name)
    return base_command.format(user_name=safe_user)
