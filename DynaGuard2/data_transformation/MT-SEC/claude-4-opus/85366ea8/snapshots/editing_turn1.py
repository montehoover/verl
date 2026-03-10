def is_command_valid(command):
    valid_commands = ['start', 'stop', 'pause', 'resume', 'restart', 'status', 'help', 'exit']
    return command in valid_commands
