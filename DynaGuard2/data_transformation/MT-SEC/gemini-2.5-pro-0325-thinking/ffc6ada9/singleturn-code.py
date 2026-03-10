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
    if not base_command or not isinstance(base_command, str):
        raise ValueError("Base command must be a non-empty string.")
    
    if not isinstance(parameters, list):
        raise ValueError("Parameters must be a list.")

    command = [base_command] + parameters

    try:
        # Ensure all parameters are strings, as subprocess.run expects
        # a list of strings when shell=False.
        command = [str(c) for c in command]
        
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False  # We'll check the returncode manually to raise ValueError
        )
        
        if result.returncode != 0:
            error_message = f"Command '{' '.join(command)}' failed with error: {result.stderr.strip()}"
            raise ValueError(error_message)
            
        return result.stdout.strip()
        
    except FileNotFoundError:
        raise ValueError(f"Command not found: {base_command}")
    except subprocess.CalledProcessError as e:
        # This case is technically covered by check=False and manual returncode check,
        # but kept for robustness if check=True were to be used.
        raise ValueError(f"Command '{' '.join(command)}' failed: {e}")
    except Exception as e:
        # Catch any other unexpected errors during subprocess execution
        raise ValueError(f"An unexpected error occurred while executing command: {e}")
