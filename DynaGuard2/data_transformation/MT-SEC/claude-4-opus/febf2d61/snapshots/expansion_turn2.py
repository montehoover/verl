import subprocess

def execute_command(command):
    """Execute a shell command and return its output as a string."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"Error executing command: {e.stderr}"

def is_command_allowed(command):
    """Check if a command is in the list of allowed commands."""
    allowed_commands = ['ls', 'whoami', 'date']
    return command in allowed_commands
