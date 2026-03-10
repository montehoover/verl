import subprocess

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
    # Example usage:
    output = execute_command("ls -l")
    print("Output of 'ls -l':")
    print(output)

    output_error = execute_command("ls non_existent_file")
    print("\nOutput of 'ls non_existent_file':")
    print(output_error)
