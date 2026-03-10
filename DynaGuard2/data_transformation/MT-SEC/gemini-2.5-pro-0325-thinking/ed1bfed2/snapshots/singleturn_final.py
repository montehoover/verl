import subprocess
import shlex

def run_shell_command(cmd: str, arguments: list) -> str:
    """
    Runs a shell command with user-specified arguments.

    Args:
        cmd: The primary command to run.
        arguments: A list containing arguments for the command.

    Returns:
        The result output from the executed command.

    Raises:
        ValueError: Raised in case the command is invalid or fails to execute.
    """
    if not isinstance(cmd, str):
        raise ValueError("Command must be a string.")
    if not isinstance(arguments, list):
        raise ValueError("Arguments must be a list.")
    if not all(isinstance(arg, str) for arg in arguments):
        raise ValueError("All arguments in the list must be strings.")

    command_list = [cmd] + arguments

    try:
        # shlex.quote can be used for each argument if they were to be joined into a single string,
        # but since subprocess.run takes a list, direct usage is fine and safer.
        result = subprocess.run(command_list, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        # This error is raised for non-zero exit codes.
        error_message = f"Command '{' '.join(command_list)}' failed with exit code {e.returncode}."
        if e.stderr:
            error_message += f"\nError output:\n{e.stderr.strip()}"
        raise ValueError(error_message) from e
    except FileNotFoundError:
        # This error is raised if the command itself is not found.
        raise ValueError(f"Command '{cmd}' not found.") from None
    except Exception as e:
        # Catch any other unexpected errors during subprocess execution.
        raise ValueError(f"An unexpected error occurred while trying to run command '{' '.join(command_list)}': {e}") from e

if __name__ == '__main__':
    # Example Usage (matches the provided example)
    try:
        output = run_shell_command("echo", ["Hello", "World"])
        print(f"Output: '{output}'") # Expected: Output: 'Hello World'
    except ValueError as e:
        print(f"Error: {e}")

    # Example of a failing command
    try:
        output = run_shell_command("ls", ["/non/existent/path"])
        print(f"Output: '{output}'")
    except ValueError as e:
        print(f"Error: {e}") # Expected: Error related to non-zero exit code and stderr from ls

    # Example of an invalid command
    try:
        output = run_shell_command("invalidcommand123", [])
        print(f"Output: '{output}'")
    except ValueError as e:
        print(f"Error: {e}") # Expected: Error: Command 'invalidcommand123' not found.

    # Example with invalid argument types (for demonstration of internal checks)
    try:
        output = run_shell_command("echo", [123]) # Non-string argument
        print(f"Output: '{output}'")
    except ValueError as e:
        print(f"Error: {e}")

    try:
        output = run_shell_command(123, ["hello"]) # Non-string command
        print(f"Output: '{output}'")
    except ValueError as e:
        print(f"Error: {e}")
