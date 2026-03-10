import shlex

def build_command_string(base_command: str, args: list[str]) -> str:
    """
    Constructs a command string from a base command and a list of arguments.

    Args:
        base_command: The main command to execute (e.g., "ls", "git").
        args: A list of arguments for the command (e.g., ["-l", "/tmp"]).

    Returns:
        A string representing the full command, with arguments safely quoted.
    """
    if not isinstance(base_command, str):
        raise TypeError("base_command must be a string")
    if not isinstance(args, list):
        raise TypeError("args must be a list")
    if not all(isinstance(arg, str) for arg in args):
        raise TypeError("all elements in args must be strings")

    command_parts = [shlex.quote(base_command)]
    for arg in args:
        command_parts.append(shlex.quote(arg))
    
    return " ".join(command_parts)

if __name__ == '__main__':
    # Example Usage
    cmd = "ls"
    arguments = ["-l", "/home/user/Documents", "file with spaces.txt"]
    command_string = build_command_string(cmd, arguments)
    print(f"Generated command string: {command_string}")

    cmd2 = "echo"
    arguments2 = ["Hello, world!", "Here's a quote: \" and '", "$HOME"]
    command_string2 = build_command_string(cmd2, arguments2)
    print(f"Generated command string: {command_string2}")

    try:
        build_command_string(123, ["arg1"])
    except TypeError as e:
        print(f"Caught expected error: {e}")

    try:
        build_command_string("cmd", "not a list")
    except TypeError as e:
        print(f"Caught expected error: {e}")
    
    try:
        build_command_string("cmd", ["arg1", 123])
    except TypeError as e:
        print(f"Caught expected error: {e}")
