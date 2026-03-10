import subprocess

def call_system_command(options=None):
    """
    Executes the shell command "ls" with the given options and returns its output.

    Args:
        options (list, optional): A list of options for the "ls" command.
                                  Defaults to None, which means no options.

    Returns:
        str: The standard output of the command.

    Raises:
        ValueError: If the command execution fails.
    """
    if options is None:
        options = []

    command = ["ls"] + options
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
        # This handles the case where 'ls' command itself is not found
        raise ValueError(f"Command 'ls' not found: {e}") from e

if __name__ == '__main__':
    try:
        print("--- Running 'ls -l' ---")
        output_l = call_system_command(options=["-l"])
        print("Command output:")
        print(output_l)

        print("\n--- Running 'ls -a' ---")
        output_a = call_system_command(options=["-a"])
        print("Command output:")
        print(output_a)

        print("\n--- Running 'ls -l -a' ---")
        output_la = call_system_command(options=["-l", "-a"])
        print("Command output:")
        print(output_la)

        print("\n--- Running 'ls' (no options) ---")
        output_none = call_system_command()
        print("Command output:")
        print(output_none)

        print("\n--- Running 'ls' with an invalid option (should fail) ---")
        try:
            call_system_command(options=["--invalid-option"])
        except ValueError as e_invalid:
            print(f"Caught expected error: {e_invalid}")

    except ValueError as e:
        print(f"An unexpected error occurred: {e}")
