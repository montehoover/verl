import subprocess

def run_shell_command(base_command, arguments):
    """
    Execute a shell command with arguments and return the standard output.
    
    Args:
        base_command (str): The base command (e.g., "ls")
        arguments (list): List of arguments (e.g., ["-l", "/home"])
    
    Returns:
        str: The standard output from the command
    """
    full_command = [base_command] + arguments
    result = subprocess.run(full_command, capture_output=True, text=True)
    return result.stdout
