import subprocess
import shlex

def run_command(cmd: str, cmd_arguments: list) -> str:
    """
    Runs a shell command with parameters provided by the user.

    Args:
        cmd: The main command that needs to be executed.
        cmd_arguments: A collection of arguments to pass alongside the command.

    Returns:
        The output produced by the executed shell command.

    Raises:
        ValueError: Raised when the command provided is incorrect or cannot be processed.
    """
    if not cmd:
        raise ValueError("Command cannot be empty.")

    command_parts = [cmd] + [shlex.quote(str(arg)) for arg in cmd_arguments]
    
    # Although the problem description implies cmd_arguments are already strings,
    # shlex.quote(str(arg)) ensures they are properly quoted for shell safety.
    # However, for subprocess.run with a list of arguments (shell=False, which is default and safer),
    # direct quoting of arguments is not strictly necessary as they are passed directly to the executable.
    # The primary benefit of shlex.quote here would be if we were constructing a single command string
    # to be passed to shell=True, which is generally discouraged.
    # For direct execution (shell=False), simply using [cmd] + cmd_arguments is often sufficient
    # if cmd_arguments are guaranteed to be strings.
    # Let's stick to a safer direct list passing without shlex.quote on arguments for shell=False.
    # The command itself (cmd) should be a simple command name. If it could contain spaces
    # or needs complex parsing, shlex.split() might be used on the initial command string.

    full_command_list = [cmd] + cmd_arguments

    try:
        # Execute the command
        # capture_output=True captures stdout and stderr
        # text=True decodes stdout and stderr as text (usually UTF-8)
        # check=True raises CalledProcessError if the command returns a non-zero exit status
        result = subprocess.run(full_command_list, capture_output=True, text=True, check=True)
        # Return the standard output, stripping any trailing newline
        return result.stdout.strip()
    except FileNotFoundError:
        # This occurs if the command executable is not found
        raise ValueError(f"Command not found: {cmd}")
    except subprocess.CalledProcessError as e:
        # This occurs if the command returns a non-zero exit status
        # e.stderr can be useful for debugging
        error_message = f"Command '{' '.join(full_command_list)}' failed with exit code {e.returncode}."
        if e.stderr:
            error_message += f"\nError output:\n{e.stderr.strip()}"
        raise ValueError(error_message)
    except Exception as e:
        # Catch any other unexpected errors during subprocess execution
        raise ValueError(f"An unexpected error occurred while running the command: {str(e)}")

if __name__ == '__main__':
    # Example Usage based on the problem description
    try:
        output = run_command("echo", ["Hello", "World"])
        print(f"Output: '{output}'") # Expected: Output: 'Hello World'
    except ValueError as e:
        print(f"Error: {e}")

    try:
        # Example of a failing command (ls on a non-existent file)
        output = run_command("ls", ["non_existent_file_for_testing_error"])
        print(f"Output: '{output}'")
    except ValueError as e:
        print(f"Error: {e}") # Expected: Error related to ls failing

    try:
        # Example of a command not found
        output = run_command("nonexistentcommand123", [])
        print(f"Output: '{output}'")
    except ValueError as e:
        print(f"Error: {e}") # Expected: Error: Command not found: nonexistentcommand123
    
    try:
        # Example with empty command
        output = run_command("", ["arg1"])
        print(f"Output: '{output}'")
    except ValueError as e:
        print(f"Error: {e}") # Expected: Error: Command cannot be empty.
