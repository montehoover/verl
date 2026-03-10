def run_shell_command(base_command, arguments=None):
    """
    Assemble a shell command from a base command string and a list of arguments.

    Example:
        run_shell_command("ls", ["-l", "/home"]) -> "ls -l /home"

    :param base_command: The base command as a string.
    :param arguments: A list of arguments to append to the command.
    :return: The full command as a single string.
    """
    if arguments is None:
        arguments = []
    return " ".join([str(base_command)] + [str(arg) for arg in arguments])
