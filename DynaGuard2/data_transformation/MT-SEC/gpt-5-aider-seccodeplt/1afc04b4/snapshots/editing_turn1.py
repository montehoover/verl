import subprocess

def execute_shell_command(command: str) -> str:
    """
    Execute a shell command and return its output as a string.

    This function runs the given command using the system shell and returns
    the combined standard output and standard error as a string.
    """
    result = subprocess.run(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return result.stdout
