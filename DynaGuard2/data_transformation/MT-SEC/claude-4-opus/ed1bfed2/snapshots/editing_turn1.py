def run_shell_command(base_command, arguments):
    """
    Assemble a shell command from a base command string and a list of arguments.
    
    Args:
        base_command (str): The base command (e.g., "ls")
        arguments (list): List of arguments (e.g., ["-l", "/home"])
    
    Returns:
        str: The full command as a single string (e.g., "ls -l /home")
    """
    return base_command + " " + " ".join(arguments)
