def is_command_allowed(command):
    allowed_commands = ['ls', 'whoami', 'date']
    if command in allowed_commands:
        return 'Command is allowed'
    else:
        return 'Command is not allowed'
