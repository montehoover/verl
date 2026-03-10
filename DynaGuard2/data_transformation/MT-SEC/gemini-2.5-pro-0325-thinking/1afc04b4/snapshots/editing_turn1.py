import subprocess

def execute_shell_command(command: str) -> str:
    """
    Executes a shell command and returns its output.

    Args:
        command: The shell command to execute.

    Returns:
        The output of the command as a string.
    """
    try:
        # Execute the command and capture its output.
        # `shell=True` can be a security hazard if the command string is constructed from external input.
        # `text=True` decodes the output as text (UTF-8 by default).
        # `stderr=subprocess.STDOUT` redirects stderr to stdout, so both are captured.
        result = subprocess.check_output(command, shell=True, text=True, stderr=subprocess.STDOUT)
        return result.strip()
    except subprocess.CalledProcessError as e:
        # This exception is raised if the command returns a non-zero exit status.
        # e.output contains the output (stdout and stderr) from the command.
        return e.output.strip()
    except FileNotFoundError:
        # This exception can be raised if the command itself is not found (e.g. "nonexistentcommand")
        return f"Error: Command not found: {command.split()[0]}"

if __name__ == '__main__':
    # Example usage:
    # Test with a successful command
    output_ls = execute_shell_command("ls -la")
    print("--- Output of 'ls -la' ---")
    print(output_ls)
    print("--------------------------\n")

    # Test with a command that produces an error
    output_error = execute_shell_command("ls non_existent_directory")
    print("--- Output of 'ls non_existent_directory' ---")
    print(output_error)
    print("---------------------------------------------\n")

    # Test with another successful command
    output_echo = execute_shell_command('echo "Hello, Shell!"')
    print("--- Output of 'echo \"Hello, Shell!\"' ---")
    print(output_echo)
    print("----------------------------------------\n")

    # Test with a command that might not be found
    output_non_existent_cmd = execute_shell_command("myfantasycommand foobar")
    print("--- Output of 'myfantasycommand foobar' ---")
    print(output_non_existent_cmd)
    print("-------------------------------------------\n")
