# List of commands that users are allowed to execute
ALLOWED_COMMANDS = ['ls', 'whoami', 'date']


def check_and_execute_command(usr_cmd):
    """
    Validate if a user command is in the list of allowed commands.
    
    This function checks whether the provided command string matches
    one of the predefined safe commands that users are permitted to run.
    
    Args:
        usr_cmd (str): The user's input command to be validated.
        
    Returns:
        str or None: The input command string if it's valid, None otherwise.
    """
    if usr_cmd in ALLOWED_COMMANDS:
        return usr_cmd
    else:
        return None
