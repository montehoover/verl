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
    # Extract the base command (e.g., 'ls' from 'ls -l')
    base_command = command.split()[0]
    return base_command in ALLOWED_COMMANDS

def execute_command(command: str) -> str:
    """
    Executes a shell command and returns its output.

    Args:
        command: The command to execute.

    Returns:
        The standard output of the command as a string.
    """
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        # Optionally, you could raise the exception or return stderr
        # For now, returning a message indicating the error.
        return f"Error executing command: {e}\nStderr: {e.stderr}"

if __name__ == '__main__':
    # Example usage for is_command_allowed:
    print(f"Is 'ls -l' allowed? {is_command_allowed('ls -l')}")
    print(f"Is 'whoami' allowed? {is_command_allowed('whoami')}")
    print(f"Is 'date' allowed? {is_command_allowed('date')}")
    print(f"Is 'rm -rf /' allowed? {is_command_allowed('rm -rf /')}")
    print(f"Is 'git status' allowed? {is_command_allowed('git status')}")


    # Example usage for execute_command (demonstrating allowed commands):
    commands_to_test = ["ls -l", "whoami", "date", "pwd"] # pwd is not in ALLOWED_COMMANDS

    for cmd in commands_to_test:
        print(f"\nTesting command: {cmd}")
        if is_command_allowed(cmd):
            output = execute_command(cmd)
            print(f"Output of '{cmd}':")
            print(output)
        else:
            print(f"Command '{cmd}' is not allowed.")

    output_error = execute_command("ls non_existent_file") # This will still run if 'ls' is allowed
    print("\nOutput of 'ls non_existent_file' (if 'ls' is allowed):")
    print(output_error)
