import shlex
import subprocess

def construct_command(base_command: str, parameters: list[str]) -> str:
    """
    Constructs a shell command from a base command and a list of parameters.

    Args:
        base_command: The base command string (e.g., "ls", "git").
        parameters: A list of strings, where each string is a parameter
                    or an argument to the command.

    Returns:
        A single string representing the full command, with parameters
        appropriately quoted if they contain spaces or special characters.
    """
    if not isinstance(base_command, str):
        raise TypeError("base_command must be a string.")
    if not base_command.strip():
        raise ValueError("base_command cannot be empty or whitespace.")
    if not isinstance(parameters, list):
        raise TypeError("parameters must be a list.")
    if not all(isinstance(param, str) for param in parameters):
        raise TypeError("All elements in parameters must be strings.")

    # shlex.quote will handle spaces and special characters in parameters
    quoted_parameters = [shlex.quote(param) for param in parameters]
    full_command = base_command
    if quoted_parameters:
        full_command += " " + " ".join(quoted_parameters)
    return full_command


def parse_command(command_string: str) -> list[str]:
    """
    Parses a full command string into a list of command components.

    Args:
        command_string: The full command string to parse.

    Returns:
        A list of strings, where each string is a component of the command,
        parsed with consideration for quotes and special characters.
    """
    if not isinstance(command_string, str):
        raise TypeError("command_string must be a string.")
    if not command_string.strip():
        raise ValueError("command_string cannot be empty or whitespace.")
    
    return shlex.split(command_string)


def execute_command(base_command: str, parameters: list[str]) -> str:
    """
    Executes a shell command and returns its output.

    Args:
        base_command: The base command to execute (e.g., "ls", "echo").
        parameters: A list of strings, where each string is an argument
                    to the command.

    Returns:
        The standard output of the command as a string.

    Raises:
        TypeError: If base_command is not a string or parameters is not a list
                   of strings.
        ValueError: If base_command is empty, or if the command execution fails
                    (e.g., command not found, non-zero exit code).
    """
    if not isinstance(base_command, str):
        raise TypeError("base_command must be a string.")
    if not base_command.strip():
        raise ValueError("base_command cannot be empty or whitespace.")
    if not isinstance(parameters, list):
        raise TypeError("parameters must be a list.")
    if not all(isinstance(param, str) for param in parameters):
        raise TypeError("All elements in parameters must be strings.")

    command_parts = [base_command] + parameters

    try:
        # Using shlex.join for constructing the command string for error messages
        # is safer than simple join, though command_parts is already a list of strings.
        # For execution, subprocess.run takes a list of arguments directly.
        process = subprocess.run(
            command_parts,
            capture_output=True,
            text=True,
            check=True  # Raises CalledProcessError for non-zero exit codes
        )
        return process.stdout.strip()
    except FileNotFoundError:
        # Command not found
        raise ValueError(f"Command not found: {base_command}")
    except subprocess.CalledProcessError as e:
        # Command returned a non-zero exit code
        error_message = f"Command '{shlex.join(command_parts)}' failed with exit code {e.returncode}."
        if e.stderr:
            error_message += f"\nStderr:\n{e.stderr.strip()}"
        raise ValueError(error_message)
    except Exception as e:
        # Catch any other unexpected errors during subprocess execution
        raise ValueError(f"An unexpected error occurred while executing command '{shlex.join(command_parts)}': {str(e)}")


if __name__ == '__main__':
    # Example Usage
    cmd1 = construct_command("ls", ["-l", "-a", "/tmp/my folder"])
    print(f"Command 1: {cmd1}")

    cmd2 = construct_command("git", ["commit", "-m", "Initial commit with spaces"])
    print(f"Command 2: {cmd2}")

    cmd3 = construct_command("echo", ["Hello, world!"])
    print(f"Command 3: {cmd3}")

    cmd4 = construct_command("find", [".", "-name", "*.txt"])
    print(f"Command 4: {cmd4}")

    cmd5 = construct_command("my_command", [])
    print(f"Command 5: {cmd5}")

    # Example of error handling (optional to run)
    try:
        construct_command(" ", ["param1"])
    except ValueError as e:
        print(f"Error example 1: {e}")

    try:
        construct_command("cmd", [123]) # type: ignore
    except TypeError as e:
        print(f"Error example 2: {e}")

    # Example Usage for parse_command
    parsed_cmd1 = parse_command("ls -l -a '/tmp/my folder'")
    print(f"Parsed Command 1: {parsed_cmd1}")

    parsed_cmd2 = parse_command("git commit -m 'Initial commit with spaces'")
    print(f"Parsed Command 2: {parsed_cmd2}")

    parsed_cmd3 = parse_command("echo 'Hello, world!'")
    print(f"Parsed Command 3: {parsed_cmd3}")
    
    parsed_cmd4 = parse_command("find . -name '*.txt'")
    print(f"Parsed Command 4: {parsed_cmd4}")

    # Example of error handling for parse_command (optional to run)
    try:
        parse_command("   ")
    except ValueError as e:
        print(f"Error example (parse_command) 1: {e}")
    
    try:
        parse_command(123) # type: ignore
    except TypeError as e:
        print(f"Error example (parse_command) 2: {e}")

    # Example Usage for execute_command
    print("\n--- execute_command Examples ---")
    try:
        output1 = execute_command("echo", ["Hello", "World from execute_command"])
        print(f"Execute Command 1 Output: '{output1}'")
    except ValueError as e:
        print(f"Execute Command 1 Error: {e}")

    try:
        # Assuming 'ls' is available on the system
        output2 = execute_command("ls", ["-lha"]) # This will list files in the current dir
        # To avoid printing a potentially long list, we'll just confirm it ran.
        # print(f"Execute Command 2 Output:\n{output2}")
        print(f"Execute Command 2 (ls -lha) ran successfully. Output length: {len(output2)}")
    except ValueError as e:
        print(f"Execute Command 2 Error: {e}")
    except FileNotFoundError: # ls might not be on all systems, though common
        print("Execute Command 2 Error: 'ls' command not found.")


    try:
        output3 = execute_command("non_existent_command_gfdgfdg", ["-arg"])
        print(f"Execute Command 3 Output: {output3}")
    except ValueError as e:
        print(f"Execute Command 3 Error: {e}")

    try:
        # Command that exists but fails
        output4 = execute_command("git", ["non_existent_git_subcommand"])
        print(f"Execute Command 4 Output: {output4}")
    except ValueError as e:
        print(f"Execute Command 4 Error: {e}")
    
    try:
        execute_command("", ["param"])
    except ValueError as e:
        print(f"Error example (execute_command) 1: {e}")

    try:
        execute_command("cmd", [123]) # type: ignore
    except TypeError as e:
        print(f"Error example (execute_command) 2: {e}")
