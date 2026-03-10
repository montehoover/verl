import subprocess

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
    try:
        # Test with a successful command
        print("Executing 'ls -l /dev/null':")
        output = execute_command("ls -l /dev/null")
        print("Output:\n", output)
        print("-" * 20)

        # Test with a command that produces output
        print("Executing 'echo Hello World':")
        output_echo = execute_command("echo Hello World from execute_command")
        print("Output:\n", output_echo)
        print("-" * 20)

    except RuntimeError as e:
        print(f"Error during example execution: {e}")
        print("-" * 20)

    try:
        # Test with a failing command (non-existent command)
        print("Executing 'non_existent_command':")
        output_fail = execute_command("non_existent_command")
        print("Output (should not reach here):\n", output_fail)
    except RuntimeError as e:
        print(f"Caught expected error: {e}")
    finally:
        print("-" * 20)

    try:
        # Test with a command that returns non-zero exit code
        print("Executing 'ls /non_existent_directory':")
        output_fail_dir = execute_command("ls /non_existent_directory")
        print("Output (should not reach here):\n", output_fail_dir)
    except RuntimeError as e:
        print(f"Caught expected error: {e}")
    finally:
        print("-" * 20)
