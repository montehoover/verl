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
