def validate_command(command):
    valid_commands = ['start', 'stop', 'pause', 'resume', 'restart', 'status', 'help', 'exit']
    if command in valid_commands:
        return command
    return None
