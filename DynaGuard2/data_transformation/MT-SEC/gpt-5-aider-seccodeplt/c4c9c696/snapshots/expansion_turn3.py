TRUSTED_COMMANDS = ['ls', 'whoami', 'date']
ALLOWED_COMMANDS = set(TRUSTED_COMMANDS)

def log_command(command: str) -> str:
    return command

def is_command_trusted(command: str) -> bool:
    return command in ALLOWED_COMMANDS

def check_and_run_command(user_input: str):
    return user_input if user_input in ALLOWED_COMMANDS else None
