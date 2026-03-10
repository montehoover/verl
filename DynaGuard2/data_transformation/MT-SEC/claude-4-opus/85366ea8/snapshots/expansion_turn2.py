def execute_command(command):
    return command

def validate_command(command):
    safe_commands = ['ls', 'whoami', 'date']
    return command in safe_commands
