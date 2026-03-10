import subprocess

def run_shell_command(base_command, arguments=None):
    """
    Execute a shell command composed of a base command and a list of arguments,
    returning the standard output as a string.

    Example:
        run_shell_command("ls", ["-l", "/home"]) -> "<directory listing>"

    :param base_command: The base command as a string.
    :param arguments: A list of arguments to append to the command.
    :return: The standard output of the command as a string.
    :raises subprocess.CalledProcessError: If the command exits with a non-zero status.
    """
    if arguments is None:
        arguments = []
    cmd = [str(base_command)] + [str(arg) for arg in arguments]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return result.stdout
