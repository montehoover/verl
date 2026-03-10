import subprocess
import shlex

def call_system_command(sys_command, params=None):
    """
    Executes a user-specified shell command with a list of arguments.

    Args:
        sys_command (str): The primary command to run (e.g., "ls", "echo").
        params (list, optional): A list of arguments for the command.
                                 Defaults to None, which means no arguments.

    Returns:
        str: The standard output of the command.

    Raises:
        ValueError: If the command execution fails.
    """
    if params is None:
        params = []

    # Ensure sys_command is a single command, not multiple commands or with shell metacharacters
    # shlex.split can help if sys_command might come from untrusted input,
    # but here we assume sys_command is the command itself.
    # For simplicity, we'll assume sys_command is just the command name.
    # If sys_command could be "ls -l", it should be split, or passed as "ls" with params=["-l"]
    # The current request implies sys_command is "ls" and params is ["-l"].

    command = [sys_command] + params
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True  # This will raise CalledProcessError for non-zero exit codes
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        raise ValueError(f"Command '{' '.join(e.cmd)}' failed with error: {e.stderr}") from e
    except FileNotFoundError as e:
        # This handles the case where the command itself is not found
        raise ValueError(f"Command '{sys_command}' not found: {e}") from e

if __name__ == '__main__':
    try:
        print("--- Running 'ls -l' ---")
        output_ls_l = call_system_command(sys_command="ls", params=["-l"])
        print("Command output:")
        print(output_ls_l)

        print("\n--- Running 'echo Hello World' ---")
        output_echo = call_system_command(sys_command="echo", params=["Hello", "World"])
        print("Command output:")
        print(output_echo)

        print("\n--- Running 'pwd' (no params) ---")
        output_pwd = call_system_command(sys_command="pwd")
        print("Command output:")
        print(output_pwd)

        print("\n--- Running a command that should fail (e.g., 'ls --invalid-option') ---")
        try:
            call_system_command(sys_command="ls", params=["--invalid-option"])
        except ValueError as e_invalid:
            print(f"Caught expected error: {e_invalid}")

        print("\n--- Running a non-existent command (should fail) ---")
        try:
            call_system_command(sys_command="nonexistentcommand123")
        except ValueError as e_nonexistent:
            print(f"Caught expected error: {e_nonexistent}")

    except ValueError as e:
        print(f"An unexpected error occurred: {e}")
