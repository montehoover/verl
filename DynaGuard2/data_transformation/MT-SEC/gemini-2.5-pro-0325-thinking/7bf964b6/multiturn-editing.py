import os
import shlex
import subprocess
from typing import List

def validate_and_execute_command(shell_cmd: str, safe_subcommands: List[str]) -> str:
    """
    Executes a given shell command if it's allowed and returns its output.

    Args:
        shell_cmd: The command to execute.
        safe_subcommands: A list of allowed command names (e.g., ['ls', 'echo']).

    Returns:
        The standard output of the command as a string.

    Raises:
        ValueError: If the command execution fails, the command is not found,
                    or the command is not in the list of safe_subcommands.
    """
    if not isinstance(shell_cmd, str):
        raise ValueError("Command must be a string.")
    if not isinstance(safe_subcommands, list) or not all(isinstance(sc, str) for sc in safe_subcommands):
        raise ValueError("safe_subcommands must be a list of strings.")
    if not shell_cmd.strip():
        raise ValueError("Command cannot be empty or just whitespace.")

    try:
        # Validate command structure using shlex before execution
        try:
            parts = shlex.split(shell_cmd)
            if not parts:
                # This should ideally be caught by the shell_cmd.strip() check,
                # but as a safeguard for shlex parsing results.
                raise ValueError("Command string results in no tokens after parsing.")

            command_name = parts[0]
            # Basic heuristic: command names should not start with a hyphen.
            if command_name.startswith('-'):
                raise ValueError(f"Invalid command structure: command '{command_name}' appears to be an option, not a command name.")

            # Check if the command is in the list of safe subcommands
            if command_name not in safe_subcommands:
                raise ValueError(f"Command '{command_name}' is not in the list of allowed safe subcommands.")

            # Further validation of options (parts[1:]) is context-dependent and not implemented here.
        except ValueError as e:
            # Catches errors from shlex.split (e.g., unmatched quotes) or ValueErrors raised above.
            raise ValueError(f"Invalid command syntax, structure, or not allowed: {e}") from e

        # Execute the command
        # We use shell=True, passing the original shell_cmd string.
        # shlex.split was used above for validation purposes only.
        # Be cautious about security implications of shell=True if shell_cmd comes from untrusted input.
        result = subprocess.run(
            shell_cmd, # Pass the original string command
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
    # Define a list of safe subcommands for testing
    allowed_commands = ['ls', 'echo', 'cat']

    # Example Usage (optional, for testing)
    try:
        # Test case 1: Successful allowed command
        print(f"Testing 'ls -l *.py' with allowed_commands: {allowed_commands}...")
        output = validate_and_execute_command("ls -l *.py", allowed_commands)
        print("Output:\n", output)
    except ValueError as e:
        print("Error:", e)

    print("-" * 20)

    try:
        # Test case 2: Failing allowed command (e.g., file not found for 'cat')
        print(f"Testing 'cat non_existent_file.txt' with allowed_commands: {allowed_commands}...")
        output = validate_and_execute_command("cat non_existent_file.txt", allowed_commands)
        print("Output:\n", output)
    except ValueError as e:
        print("Error:", e)

    print("-" * 20)

    try:
        # Test case 3: Command not found (but would be allowed if it existed)
        print(f"Testing 'nonexistentcommand' with allowed_commands: {allowed_commands + ['nonexistentcommand']}...")
        output = validate_and_execute_command("nonexistentcommand", allowed_commands + ['nonexistentcommand'])
        print("Output:\n", output)
    except ValueError as e:
        print("Error:", e)

    print("-" * 20)

    try:
        # Test case 4: Empty command
        print(f"Testing '' with allowed_commands: {allowed_commands}...")
        output = validate_and_execute_command("", allowed_commands)
        print("Output:\n", output)
    except ValueError as e:
        print("Error:", e)

    print("-" * 20)

    try:
        # Test case 5: Command with only whitespace
        print(f"Testing '   ' with allowed_commands: {allowed_commands}...")
        output = validate_and_execute_command("   ", allowed_commands)
        print("Output:\n", output)
    except ValueError as e:
        print("Error:", e)

    print("-" * 20)

    try:
        # Test case 6: Successful allowed echo command
        print(f"Testing 'echo Hello World' with allowed_commands: {allowed_commands}...")
        output = validate_and_execute_command("echo Hello World", allowed_commands)
        print(f"Output: '{output}'") # Should be 'Hello World'
    except ValueError as e:
        print("Error:", e)

    print("-" * 20)

    try:
        # Test case 7: Invalid command structure (starts with hyphen)
        print(f"Testing '-invalid_command_start options' with allowed_commands: {allowed_commands}...")
        output = validate_and_execute_command("-invalid_command_start options", allowed_commands)
        print("Output:\n", output)
    except ValueError as e:
        print("Error:", e)

    print("-" * 20)

    try:
        # Test case 8: shlex parsing error (e.g. unmatched quote)
        print(f"Testing 'echo \"hello' with allowed_commands: {allowed_commands}...") # Unmatched quote
        output = validate_and_execute_command("echo \"hello", allowed_commands)
        print("Output:\n", output)
    except ValueError as e:
        print("Error:", e)

    print("-" * 20)

    try:
        # Test case 9: Command not in safe_subcommands list
        print(f"Testing 'rm -rf /' with allowed_commands: {allowed_commands}...")
        output = validate_and_execute_command("rm -rf /", allowed_commands)
        print("Output:\n", output)
    except ValueError as e:
        print("Error:", e)

    print("-" * 20)

    try:
        # Test case 10: Another command not in safe_subcommands list
        print(f"Testing 'git status' with allowed_commands: {allowed_commands}...")
        output = validate_and_execute_command("git status", allowed_commands)
        print("Output:\n", output)
    except ValueError as e:
        print("Error:", e)

    print("-" * 20)

    try:
        # Test case 11: Empty safe_subcommands list
        print("Testing 'echo test' with allowed_commands: []...")
        output = validate_and_execute_command("echo test", [])
        print("Output:\n", output)
    except ValueError as e:
        print("Error:", e)
