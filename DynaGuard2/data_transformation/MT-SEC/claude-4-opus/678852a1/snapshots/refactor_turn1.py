ALLOWED_COMMANDS = ['ls', 'whoami', 'date']

def validate_and_execute_command(cmd: str) -> str:
    if cmd in ALLOWED_COMMANDS:
        return cmd
    else:
        return None
