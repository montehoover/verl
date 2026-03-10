ALLOWED_COMMANDS = ['ls', 'whoami', 'date']

def verify_and_run_command(command):
    if command in ALLOWED_COMMANDS:
        return command
    else:
        return None
