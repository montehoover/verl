import subprocess
import datetime
import os
import shlex

def execute_shell_command(command):
    """Execute a shell command and return its output as a string."""
    # Log the command with timestamp
    with open('command_log.txt', 'a') as log_file:
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_file.write(f"[{timestamp}] {command}\n")
    
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return result.stdout

def run_shell_command_with_filters(command, allowed_subcommands_list):
    """Execute a shell command only if all subcommands are in the allowed list."""
    # Parse the command to extract subcommands
    try:
        parsed_command = shlex.split(command)
    except ValueError as e:
        return f"Error parsing command: {e}"
    
    # Check if all subcommands are allowed
    for subcommand in parsed_command:
        if subcommand not in allowed_subcommands_list:
            return f"Error: Subcommand '{subcommand}' is not allowed"
    
    # Execute the command if all subcommands are allowed
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return result.stdout
