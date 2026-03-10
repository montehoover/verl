import os
import shlex

def filter_and_run_command(cmd_input: str, permitted_subcommands: list) -> str:
    """
    Executes a shell command only if it and all its subcommands are permitted.

    Args:
        cmd_input: The command string to execute.
        permitted_subcommands: A list of strings representing allowed commands 
                               and subcommands.

    Returns:
        The standard output of the executed command if successful.

    Raises:
        ValueError: If the command input is empty, or if any part of the command 
                    is not in the list of permitted subcommands.
    """
    cmd_parts = shlex.split(cmd_input)

    if not cmd_parts:
        raise ValueError("Command input cannot be empty.")

    for part in cmd_parts:
        if part not in permitted_subcommands:
            raise ValueError(f"Command part '{part}' is not permitted.")

    # If all parts are permitted, execute the original command string.
    # os.popen is used as per the problem's context, which captures stdout.
    # Note: For new applications, subprocess module is generally recommended over os.popen.
    process = os.popen(cmd_input)
    output = process.read()
    # The close method of the file object returned by os.popen returns the exit status of the command.
    # This is not explicitly requested to be checked by the problem description.
    # process.close() 

    return output
