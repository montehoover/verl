import shlex

base_command = "bash my_script.sh {script_arg}"

def create_greeting_command(script_arg: str) -> str:
    """
    Create a shell command to run 'my_script.sh' safely with the provided argument.
    The argument is shell-escaped to prevent injection.
    """
    if not isinstance(script_arg, str):
        script_arg = str(script_arg)
    sanitized = shlex.quote(script_arg)
    return base_command.format(script_arg=sanitized)
