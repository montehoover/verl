base_command = "echo Hello, {name_of_user}!"

def assemble_greet_command(name_of_user: str) -> str:
    # Replace single quotes with '\'' to safely escape them in shell
    safe_name = name_of_user.replace("'", "'\\''")
    # Wrap the name in single quotes to prevent command injection
    return base_command.format(name_of_user=f"'{safe_name}'")
