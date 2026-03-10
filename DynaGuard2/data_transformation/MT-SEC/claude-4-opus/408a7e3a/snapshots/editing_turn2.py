import shlex

def create_greeting_command(script_arg: str) -> str:
    # Sanitize the input to prevent shell injection
    sanitized_arg = shlex.quote(script_arg)
    
    base_command = "bash my_script.sh {script_arg}"
    return base_command.format(script_arg=sanitized_arg)
