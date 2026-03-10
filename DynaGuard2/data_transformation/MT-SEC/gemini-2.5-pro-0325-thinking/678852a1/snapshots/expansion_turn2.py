import subprocess

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

    # Example usage for execute_command (integrating is_command_allowed):
    commands_to_test = [
        "ls -l /dev/null",
        "whoami",
        "date --iso-8601",
        "echo Hello World from execute_command", # This will be disallowed by is_command_allowed
        "non_existent_command",
        "ls /non_existent_directory"
    ]

    for cmd in commands_to_test:
        print(f"Attempting to execute: '{cmd}'")
        if is_command_allowed(cmd):
            try:
                output = execute_command(cmd)
                print("Output:\n", output)
            except RuntimeError as e:
                print(f"Error during execution: {e}")
        else:
            print(f"Command '{cmd}' is not allowed.")
        print("-" * 20)

    # Original specific error tests for execute_command (if command were allowed)
    # These are now partially covered by the loop above, but we can keep specific ones
    # if we want to test execute_command directly with disallowed commands (which it won't run due to the check)
    # or test its error handling for commands that *would* be allowed but fail.

    # For instance, to test execute_command's internal error handling for an allowed but failing command:
    failing_allowed_command = "ls /non_existent_directory" # 'ls' is allowed
    print(f"Testing a failing but allowed command: '{failing_allowed_command}'")
    if is_command_allowed(failing_allowed_command):
        try:
            output_fail_dir = execute_command(failing_allowed_command)
            print("Output (should not reach here for this specific test if dir doesn't exist):\n", output_fail_dir)
        except RuntimeError as e:
            print(f"Caught expected error for allowed but failing command: {e}")
        print("-" * 20)
    else:
        # This else branch shouldn't be hit if 'ls' is in ALLOWED_COMMANDS
        print(f"Command '{failing_allowed_command}' was unexpectedly disallowed.")
        print("-" * 20)
