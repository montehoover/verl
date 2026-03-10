import subprocess

def run_basic_command(command):
    """
    Runs a shell command without arguments and returns its output as a string.
    
    Args:
        command: The shell command to execute
        
    Returns:
        The command output as a string
        
    Raises:
        subprocess.CalledProcessError: If the command execution fails
    """
    result = subprocess.run(
        command,
        shell=True,
        capture_output=True,
        text=True,
        check=True
    )
    return result.stdout
