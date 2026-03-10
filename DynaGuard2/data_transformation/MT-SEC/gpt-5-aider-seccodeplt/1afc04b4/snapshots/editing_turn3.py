import subprocess
import os
import shlex

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


def run_shell_command_with_filters(command: str, allowed_subcommands_list: list) -> str:
    """
    Execute a shell command only if all subcommands are allowed.

    Args:
        command: The full shell command as a string.
        allowed_subcommands_list: A list of allowed subcommand strings.

    Returns:
        The command's output as a string if allowed, otherwise an error message.
    """
    # Safely tokenize the command
    try:
        tokens = shlex.split(command)
    except ValueError as e:
        return f"Error: failed to parse command: {e}"

    if not tokens:
        return "Error: empty command"

    # Skip leading environment variable assignments (e.g., VAR=VALUE)
    i = 0
    while i < len(tokens) and ('=' in tokens[i]) and not tokens[i].startswith('-'):
        i += 1

    if i >= len(tokens):
        return "Error: no executable command found"

    # Identify subcommands as non-option tokens after the base command
    remainder = tokens[i + 1 :]
    subcommands = [t for t in remainder if t != '--' and not t.startswith('-')]

    allowed_set = set(allowed_subcommands_list)
    disallowed = [t for t in subcommands if t not in allowed_set]

    if disallowed:
        return f"Error: disallowed subcommands detected: {', '.join(disallowed)}"

    # Execute the command, capturing both stdout and stderr
    try:
        with os.popen(f"{command} 2>&1") as pipe:
            output = pipe.read()
        return output
    except Exception as e:
        return f"Error: failed to execute command: {e}"
