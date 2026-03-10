TRUSTED_COMMANDS = {'ls', 'whoami', 'date'}

def log_command(command: str) -> str:
    return command

def is_command_trusted(command: str) -> bool:
    return command in TRUSTED_COMMANDS
