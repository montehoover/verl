ALLOWED_COMMANDS = ['ls', 'whoami', 'date']

def check_and_execute_command(usr_cmd):
    if usr_cmd in ALLOWED_COMMANDS:
        return usr_cmd
    return None
