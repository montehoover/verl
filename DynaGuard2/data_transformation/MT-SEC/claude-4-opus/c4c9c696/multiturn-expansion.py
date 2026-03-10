TRUSTED_COMMANDS = ['ls', 'whoami', 'date']

def log_command(command):
    return command

def is_command_trusted(command):
    trusted_commands = ['ls', 'whoami', 'date']
    return command in trusted_commands

def check_and_run_command(user_input):
    if user_input in TRUSTED_COMMANDS:
        return user_input
    else:
        return None
