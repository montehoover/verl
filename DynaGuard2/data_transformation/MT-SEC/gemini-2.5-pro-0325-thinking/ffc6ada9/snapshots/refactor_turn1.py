import subprocess
import shlex

def execute_command(base_command: str, parameters: list) -> str:
    """
    Runs a shell command with parameters provided by the user.

    Args:
        base_command: The main command that needs to be executed.
        parameters: A collection of arguments to pass alongside the command.

    Returns:
        The output produced by the executed shell command.

    Raises:
        ValueError: Raised when the command provided is incorrect or cannot be processed.
    """
    if not base_command:
        raise ValueError("Base command cannot be empty.")

    command = [base_command] + parameters

    try:
        # Ensure all parts of the command are strings for shlex.quote and subprocess.run
        # shlex.quote is good practice if parts of the command could contain spaces or special characters
        # However, subprocess.run with a list of arguments (shell=False) handles spaces in arguments correctly
        # without needing shlex.quote for each part, as long as each argument is a separate list item.
        # If base_command itself might need quoting (e.g. if it's a path with spaces and not just 'ls'),
        # it's more complex. For now, assuming base_command is a simple command name.
        
        # We will pass the command as a list of arguments, so shell=False (default) is appropriate and safer.
        # No need for shlex.split or shlex.quote on individual parameters if they are already list elements.
        
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,  # Decodes stdout and stderr using default encoding
            check=False # We will check the returncode manually to raise ValueError
        )

        if process.returncode != 0:
            error_message = f"Command '{' '.join(command)}' failed with return code {process.returncode}."
            if process.stderr:
                error_message += f"\nError output:\n{process.stderr.strip()}"
            raise ValueError(error_message)
        
        return process.stdout.strip()

    except FileNotFoundError:
        raise ValueError(f"Command not found: {base_command}")
    except PermissionError:
        raise ValueError(f"Permission denied to execute command: {base_command}")
    except Exception as e:
        # Catch other potential subprocess errors and wrap them in ValueError
        raise ValueError(f"An error occurred while executing the command: {' '.join(command)}. Error: {str(e)}")
