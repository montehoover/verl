import shlex

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
