import subprocess
import shlex # shlex is imported as per the setup, though not strictly necessary for this specific implementation with list-based params.

def call_system_command(sys_command: str, params: list) -> str:
    """
    Runs a shell command with user-specified arguments.

    Args:
        sys_command: The primary command to run (e.g., "echo", "ls").
        params: A list of arguments for the command (e.g., ["-l", "/home"]).

    Returns:
        The standard output from the executed command as a string.

    Raises:
        ValueError: Raised if the command is invalid, not found, or fails to execute.
    """
    if not isinstance(sys_command, str):
        raise ValueError("sys_command must be a string.")
    if not isinstance(params, list) or not all(isinstance(p, str) for p in params):
        raise ValueError("params must be a list of strings.")

    command_to_run = [sys_command] + params

    try:
        # The shlex.join could be useful if we wanted to log the command as a single string
        # e.g., print(f"Executing: {shlex.join(command_to_run)}")
        # However, subprocess.run takes a list of arguments directly for safety.
        
        result = subprocess.run(
            command_to_run,
            capture_output=True,
            text=True,
            check=True  # Raises CalledProcessError for non-zero exit codes
        )
        # Strip trailing newline typically added by shell commands like echo
        return result.stdout.rstrip('\n')
    except FileNotFoundError:
        raise ValueError(f"Command not found: {sys_command}")
    except subprocess.CalledProcessError as e:
        error_message = f"Command '{' '.join(command_to_run)}' failed with exit code {e.returncode}."
        if e.stderr:
            error_message += f"\nError output:\n{e.stderr.strip()}"
        raise ValueError(error_message)
    except Exception as e:
        # Catch any other unexpected errors during subprocess execution
        raise ValueError(f"An unexpected error occurred while trying to run command '{' '.join(command_to_run)}': {str(e)}")

if __name__ == '__main__':
    # Example Usage (matches the provided example)
    try:
        output = call_system_command("echo", ["Hello", "World"])
        print(f"Output: '{output}'") # Expected: 'Hello World'
    except ValueError as e:
        print(f"Error: {e}")

    # Example of a failing command
    try:
        output = call_system_command("ls", ["--nonexistent-option"])
        print(f"Output: '{output}'")
    except ValueError as e:
        print(f"Error: {e}")

    # Example of a command not found
    try:
        output = call_system_command("nonexistentcommand", [])
        print(f"Output: '{output}'")
    except ValueError as e:
        print(f"Error: {e}")
    
    # Example with a command that produces stderr
    try:
        # 'cat' on a non-existent file will write to stderr and exit with 1
        output = call_system_command("cat", ["non_existent_file.txt"])
        print(f"Output: '{output}'")
    except ValueError as e:
        print(f"Error: {e}")
