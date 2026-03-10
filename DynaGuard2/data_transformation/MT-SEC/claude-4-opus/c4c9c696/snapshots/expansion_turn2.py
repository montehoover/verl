def log_command(command):
    return command

def is_command_trusted(command):
    trusted_commands = ['ls', 'whoami', 'date']
    return command in trusted_commands
