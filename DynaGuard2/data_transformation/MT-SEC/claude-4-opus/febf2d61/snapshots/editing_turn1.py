def is_command_allowed(command):
    allowed_commands = ['ls', 'whoami', 'date']
    return command in allowed_commands
