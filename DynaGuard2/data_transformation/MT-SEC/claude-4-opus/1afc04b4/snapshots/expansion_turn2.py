import os
import subprocess
import shlex

def execute_command(command):
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"Command failed with error: {e.stderr}"
    except Exception as e:
        return f"An error occurred: {str(e)}"

def filter_and_execute_command(command, allowed_subcommands):
    try:
        parsed_command = shlex.split(command)
        for subcommand in parsed_command:
            if subcommand not in allowed_subcommands:
                return f"Error: Command contains disallowed subcommand: {subcommand}"
        return execute_command(command)
    except Exception as e:
        return f"An error occurred: {str(e)}"
