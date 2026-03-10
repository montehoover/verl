import os
import subprocess

def validate_and_execute_command(shell_cmd: str) -> str:
    """
    Executes a given shell command and returns its output.

    Args:
        shell_cmd: The command to execute.

    Returns:
        The standard output of the command as a string.

    Raises:
        ValueError: If the command execution fails or the command is not found.
    """
    if not isinstance(shell_cmd, str):
        raise ValueError("Command must be a string.")
    if not shell_cmd.strip():
        raise ValueError("Command cannot be empty or just whitespace.")

    try:
        # Execute the command
        # We use shell=True for simplicity here, but be cautious about security implications
        # if shell_cmd comes from untrusted input. For more complex scenarios,
        # passing a list of arguments (shell=False) is generally safer.
        result = subprocess.run(
            shell_cmd,
            shell=True,
            capture_output=True,
            text=True,
            check=False  # We will check the returncode manually
        )

        # Check if the command executed successfully
        if result.returncode != 0:
            error_message = f"Command '{shell_cmd}' failed with return code {result.returncode}."
            if result.stderr:
                error_message += f"\nError output:\n{result.stderr.strip()}"
            raise ValueError(error_message)

        return result.stdout.strip()

    except FileNotFoundError:
        # This exception occurs if the command itself is not found (e.g., 'nonexistent_command')
        # when shell=False. With shell=True, the shell usually handles this and returns a non-zero exit code.
        # However, it's good practice to handle it explicitly if it could occur.
        raise ValueError(f"Command not found: {shell_cmd.split()[0]}")
    except PermissionError:
        raise ValueError(f"Permission denied to execute: {shell_cmd.split()[0]}")
    except Exception as e:
        # Catch any other unexpected errors during subprocess execution
        raise ValueError(f"An unexpected error occurred while executing command '{shell_cmd}': {e}")

if __name__ == '__main__':
    # Example Usage (optional, for testing)
    try:
        # Test case 1: Successful command
        print("Testing 'ls -l *.py'...")
        output = validate_and_execute_command("ls -l *.py")
        print("Output:\n", output)
    except ValueError as e:
        print("Error:", e)

    print("-" * 20)

    try:
        # Test case 2: Failing command (e.g., file not found for 'cat')
        print("Testing 'cat non_existent_file.txt'...")
        output = validate_and_execute_command("cat non_existent_file.txt")
        print("Output:\n", output)
    except ValueError as e:
        print("Error:", e)

    print("-" * 20)

    try:
        # Test case 3: Command not found
        print("Testing 'nonexistentcommand'...")
        output = validate_and_execute_command("nonexistentcommand")
        print("Output:\n", output)
    except ValueError as e:
        print("Error:", e)

    print("-" * 20)

    try:
        # Test case 4: Empty command
        print("Testing ''...")
        output = validate_and_execute_command("")
        print("Output:\n", output)
    except ValueError as e:
        print("Error:", e)

    print("-" * 20)

    try:
        # Test case 5: Command with only whitespace
        print("Testing '   '...")
        output = validate_and_execute_command("   ")
        print("Output:\n", output)
    except ValueError as e:
        print("Error:", e)

    print("-" * 20)
    
    try:
        # Test case 6: Successful echo command
        print("Testing 'echo Hello World'...")
        output = validate_and_execute_command("echo Hello World")
        print(f"Output: '{output}'") # Should be 'Hello World'
    except ValueError as e:
        print("Error:", e)
