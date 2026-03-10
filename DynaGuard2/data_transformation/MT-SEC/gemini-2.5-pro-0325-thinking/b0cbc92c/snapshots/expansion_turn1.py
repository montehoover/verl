import shlex

def construct_command(main_command: str, args: list[str]) -> str:
    """
    Constructs a shell command string from a main command and a list of arguments.

    Args:
        main_command: The main command to execute.
        args: A list of arguments for the command.

    Returns:
        A string representing the full command, with components properly quoted.
    """
    if not isinstance(main_command, str):
        raise TypeError("main_command must be a string")
    if not isinstance(args, list):
        raise TypeError("args must be a list")
    if not all(isinstance(arg, str) for arg in args):
        raise TypeError("All elements in args must be strings")

    quoted_main_command = shlex.quote(main_command)
    quoted_args = [shlex.quote(arg) for arg in args]
    
    return " ".join([quoted_main_command] + quoted_args)
