TRUSTED_COMMANDS = ['ls', 'whoami', 'date']

def check_and_run_command(user_input):
    if user_input in TRUSTED_COMMANDS:
        return user_input
    else:
        return None
