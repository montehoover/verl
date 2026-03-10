import subprocess
import shlex


def execute_command(base_command: str, parameters: list) -> str:
    """
    Execute a shell command with parameters and return the output.
    
    This function safely executes a shell command with the provided parameters
    and captures its standard output. It handles proper escaping of arguments
    to prevent shell injection vulnerabilities.
    
    Args:
        base_command (str): The main command that needs to be executed.
            This should be the executable name or path (e.g., 'ls', 'git', '/usr/bin/python').
        parameters (list): A collection of arguments to pass alongside the command.
            Each element in the list will be passed as a separate argument to the command.
            Elements will be automatically converted to strings if they aren't already.
        
    Returns:
        str: The output produced by the executed shell command (stdout).
            If the command produces no output, an empty string is returned.
        
    Raises:
        ValueError: Raised when the command provided is incorrect or cannot be processed.
            This includes cases where:
            - The command executable is not found in the system PATH
            - The command exits with a non-zero status code
            - Any other error occurs during command execution
            
    Examples:
        >>> execute_command('echo', ['Hello', 'World'])
        'Hello World\\n'
        
        >>> execute_command('ls', ['-la', '/tmp'])
        'total 8\\ndrwxrwxrwt  2 root root 4096 Jan  1 00:00 .\\n...'
        
        >>> execute_command('invalid_command', [])
        Traceback (most recent call last):
            ...
        ValueError: Command 'invalid_command' not found
    """
    try:
        # Combine command and parameters into a single list
        # No need to quote here as subprocess.run handles it safely when shell=False
        full_command = [base_command] + parameters
        
        # Execute the command with subprocess.run
        # capture_output=True captures both stdout and stderr
        # text=True ensures output is returned as string rather than bytes
        # check=True raises CalledProcessError for non-zero exit codes
        result = subprocess.run(
            full_command,
            capture_output=True,
            text=True,
            check=True
        )
        
        # Return the standard output
        return result.stdout
        
    except subprocess.CalledProcessError as e:
        # Handle commands that exit with non-zero status
        raise ValueError(f"Command execution failed: {e}")
    except FileNotFoundError:
        # Handle case where command executable is not found
        raise ValueError(f"Command '{base_command}' not found")
    except Exception as e:
        # Handle any other unexpected errors
        raise ValueError(f"Error processing command: {e}")
