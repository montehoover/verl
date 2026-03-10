import subprocess
import datetime

def execute_shell_command(command):
    """Execute a shell command and return its output as a string."""
    # Log the command with timestamp
    with open('command_log.txt', 'a') as log_file:
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_file.write(f"[{timestamp}] {command}\n")
    
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return result.stdout
