import subprocess
from typing import Optional

ALLOWED_COMMANDS = ['ls', 'whoami', 'date']

def is_command_allowed(command: str) -> bool:
    """
    Checks if the given command is in the list of allowed commands.

    Args:
        command: The command string to check.

    Returns:
        True if the command is allowed, False otherwise.
    """
    # Extract the base command (the first word)
    base_command = command.strip().split()[0]
    return base_command in ALLOWED_COMMANDS

def execute_command(command: str) -> str:
    """
    Executes a shell command and returns its output.

    Args:
        command: The command string to execute.

    Returns:
        The standard output of the command as a string.

    Raises:
        RuntimeError: If the command execution fails or returns a non-zero exit code.
    """
    try:
        # Execute the command, capture output, and check for errors.
        # shell=True can be a security risk if the command string comes from untrusted input.
        # For a system utility tool where commands are controlled, it's often acceptable.
        result = subprocess.run(
            command,
            shell=True,
            check=True,  # Raises CalledProcessError for non-zero exit codes
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True  # Decodes stdout and stderr as text
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        # Handle errors specifically from the command execution (e.g., non-zero exit status)
        error_message = f"Command '{command}' failed with exit code {e.returncode}.\n"
        error_message += f"Stderr: {e.stderr.strip()}"
        raise RuntimeError(error_message) from e
    except FileNotFoundError as e:
        # Handle error if the shell or command itself is not found
        error_message = f"Command or shell not found for '{command}'. Ensure it's in PATH or provide full path."
        raise RuntimeError(error_message) from e
    except Exception as e:
        # Catch any other unexpected errors during execution
        error_message = f"An unexpected error occurred while executing command '{command}': {str(e)}"
        raise RuntimeError(error_message) from e

def validate_and_execute_command(cmd: str) -> Optional[str]:
    """
    Validates if a command is allowed and executes it if it is.

    Args:
        cmd: The command string to validate and execute.

    Returns:
        The output of the command if it's allowed and executes successfully,
        None otherwise (if not allowed or if execution fails).
    """
    if not is_command_allowed(cmd):
        # Consider logging instead of printing, or make verbosity configurable
        print(f"Info: Command '{cmd}' is not allowed.")
        return None
    try:
        output = execute_command(cmd)
        return output
    except RuntimeError as e:
        # Consider logging instead of printing
        print(f"Error: While executing allowed command '{cmd}': {e}")
        return None

if __name__ == '__main__':
    # Example usage:
    # Example usage for is_command_allowed:
    print("Checking allowed commands:")
    print(f"'ls -l /dev/null' allowed? {is_command_allowed('ls -l /dev/null')}")  # Expected: True
    print(f"'whoami' allowed? {is_command_allowed('whoami')}")              # Expected: True
    print(f"'date' allowed? {is_command_allowed('date')}")                  # Expected: True
    print(f"'rm -rf /' allowed? {is_command_allowed('rm -rf /')}")          # Expected: False
    print(f"'pwd' allowed? {is_command_allowed('pwd')}")                    # Expected: False
    print("-" * 20)

    # Define a list of commands for comprehensive testing
    commands_to_test = [
        "ls -l /dev/null",                  # Allowed, should succeed
        "whoami",                           # Allowed, should succeed
        "date --iso-8601",                  # Allowed, should succeed (if 'date' supports this option)
        "ls /non_existent_directory",       # Allowed (base 'ls'), but execution should fail
        "echo Hello World",                 # Disallowed (base 'echo' not in ALLOWED_COMMANDS)
        "non_existent_command_123",         # Disallowed (base command not in ALLOWED_COMMANDS)
        "rm -rf /",                         # Disallowed (base 'rm' not in ALLOWED_COMMANDS)
    ]

    print("\nTesting validate_and_execute_command:")
    print("=" * 40)
    for cmd_to_run in commands_to_test:
        print(f"Attempting to validate and execute: '{cmd_to_run}'")
        result = validate_and_execute_command(cmd_to_run)
        if result is not None:
            print(f"Execution successful for '{cmd_to_run}'. Output:\n{result}")
        else:
            # The validate_and_execute_command function already prints details
            # for "not allowed" or "execution error" cases.
            print(f"Execution failed or command not allowed for '{cmd_to_run}'. Result: {result}")
        print("-" * 20)
