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

def run_shell_command_with_filters(command, allowed_subcommands_list):
    try:
        if not command or not command.strip():
            raise ValueError("Invalid command: command cannot be empty")
        
        parsed_command = shlex.split(command)
        for subcommand in parsed_command:
            if subcommand not in allowed_subcommands_list:
                raise ValueError(f"Command contains disallowed subcommand: {subcommand}")
        
        result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"Command failed with error: {e.stderr}"
    except ValueError:
        raise
    except Exception as e:
        return f"An error occurred: {str(e)}"
