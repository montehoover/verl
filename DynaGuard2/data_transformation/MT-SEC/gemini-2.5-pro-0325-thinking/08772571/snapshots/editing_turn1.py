import subprocess

def call_system_command():
    """
    Executes the shell command "ls -l" and returns its output.

    Returns:
        str: The standard output of the command.

    Raises:
        ValueError: If the command execution fails.
    """
    try:
        result = subprocess.run(
            ["ls", "-l"],
            capture_output=True,
            text=True,
            check=True  # This will raise CalledProcessError for non-zero exit codes
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        raise ValueError(f"Command '{e.cmd}' failed with error: {e.stderr}") from e
    except FileNotFoundError as e:
        # This handles the case where 'ls' command itself is not found
        raise ValueError(f"Command 'ls' not found: {e}") from e

if __name__ == '__main__':
    try:
        output = call_system_command()
        print("Command output:")
        print(output)
    except ValueError as e:
        print(f"Error: {e}")
