import shlex
import os

def parse_command(command_string):
    """
    Parses a command string into a list of components using shlex.

    Args:
        command_string: The command string to parse.

    Returns:
        A list of strings, representing the components of the command.
    """
    return shlex.split(command_string)

def validate_command(command_components, permitted_subcommands):
    """
    Validates if all components of a command are in the list of permitted subcommands.

    Args:
        command_components: A list of strings, representing the parsed command.
        permitted_subcommands: A list of strings, representing allowed subcommands.

    Returns:
        A boolean: True if all components are permitted, False otherwise.
    """
    for component in command_components:
        if component not in permitted_subcommands:
            return False
    return True

def filter_and_run_command(cmd_input, permitted_subcommands):
    """
    Parses, validates, and executes a command string if it's allowed.

    Args:
        cmd_input: The command string to execute.
        permitted_subcommands: A list of strings, representing allowed subcommands/arguments.
                               Each part of the parsed command must be in this list.

    Returns:
        The standard output of the executed command if validation passes.

    Raises:
        ValueError: If the command is not valid (i.e., contains components not in
                    permitted_subcommands).
        OSError: If `os.popen` fails (e.g., command not found by the shell,
                 though this behavior can be shell-dependent). shlex.quote requires Python 3.3+.
    """
    components = parse_command(cmd_input)

    if not validate_command(components, permitted_subcommands):
        raise ValueError("Command validation failed: one or more components are not permitted.")

    # Reconstruct the command string for shell execution.
    # shlex.quote ensures each component is treated as a single argument,
    # preventing shell injection vulnerabilities if components could be crafted.
    # Requires Python 3.3+.
    command_to_execute = " ".join([shlex.quote(c) for c in components])
    
    # Execute the command using os.popen.
    # Note: subprocess.run is generally recommended over os.popen for new code
    # due to better security options and flexibility.
    pipe = os.popen(command_to_execute)
    output = pipe.read()
    # pipe.close() is called to ensure the process terminates and resources are freed.
    # The exit status can be checked here if specific handling for command execution
    # errors is required (e.g., raise RuntimeError if exit_status is non-zero).
    # For this function, we return the output regardless of the command's own exit status,
    # as long as it was validated and os.popen itself didn't fail.
    exit_status = pipe.close() 

    # Optional: Check exit_status and raise an error if the command failed.
    # if exit_status is not None and exit_status != 0:
    #     raise RuntimeError(f"Command execution failed with status {exit_status}: {command_to_execute}")
    
    return output
