import subprocess

def execute_shell_command(command: str) -> str:
    """
    Execute a shell command and return its output as a string.

    This function runs the given command using the system shell and returns
    the combined standard output and standard error as a string.
    Additionally, it logs each executed command to 'command_log.txt'.
    """
    # Log the command to the file
    try:
        with open('command_log.txt', 'a', encoding='utf-8') as log_file:
            log_file.write(f"{command}\n")
    except Exception:
        # If logging fails, continue without interrupting command execution
        pass

    result = subprocess.run(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return result.stdout
