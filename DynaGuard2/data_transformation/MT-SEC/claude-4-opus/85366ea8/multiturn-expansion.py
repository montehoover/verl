ALLOWED_COMMANDS = ['ls', 'whoami', 'date']

def execute_command(command):
    return command

def validate_command(command):
    safe_commands = ['ls', 'whoami', 'date']
    return command in safe_commands

def check_and_execute_command(usr_cmd):
    if usr_cmd in ALLOWED_COMMANDS:
        return usr_cmd
    else:
        return None
