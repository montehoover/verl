import subprocess

def execute_shell_command(command):
    """Execute a shell command and return its output as a string."""
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return result.stdout
