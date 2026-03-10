import os
import shlex

def validate_and_execute_command(shell_cmd: str, safe_subcommands: list[str]) -> str:
    """
    Executes a shell command if its main command component is in the list of safe subcommands.

    Args:
        shell_cmd: The command string to execute.
        safe_subcommands: A list of main command names (e.g., 'ls', 'echo')
                          that are considered safe to execute.

    Returns:
        The standard output (and standard error) of the executed command as a string.

    Raises:
        ValueError: If the shell_cmd is empty, syntactically invalid,
                    or if its main command component is not in safe_subcommands.
    """
    if not shell_cmd or not shell_cmd.strip():
        raise ValueError("Shell command cannot be empty.")

    try:
        # Split the command string into components. shlex handles quoted arguments.
        parsed_cmd_parts = shlex.split(shell_cmd)
    except ValueError as e:
        # This handles cases like unmatched quotes in shell_cmd.
        raise ValueError(f"Invalid command syntax: {str(e)}")

    if not parsed_cmd_parts:
        # This case might occur if shell_cmd contains only whitespace and comments
        # (if shlex comment handling were enabled, though default is off).
        # The initial `shell_cmd.strip()` check should catch most empty/whitespace-only cases.
        raise ValueError("Parsed command is empty.")

    main_command = parsed_cmd_parts[0]

    if main_command not in safe_subcommands:
        # Command is not allowed.
        error_message = f"Command '{main_command}' is not allowed."
        if safe_subcommands:
            # Use set to remove duplicates and sort for consistent error messages.
            distinct_safe_subcommands = sorted(list(set(safe_subcommands)))
            error_message += f" Permitted commands are: {', '.join(distinct_safe_subcommands)}."
        else:
            error_message += " No commands are permitted."
        raise ValueError(error_message)

    # If the command is in the safe list, execute it using os.popen.
    # Note on security: os.popen executes the command via the system's shell.
    # The entire `shell_cmd` string is passed to the shell, which will interpret
    # any shell metacharacters (e.g., ';', '&&', '$()'). This can be a security risk
    # if `shell_cmd` could contain malicious constructs beyond the `main_command`.
    # For example, if `safe_subcommands = ['echo']` and `shell_cmd = "echo hello; rm -rf /"`,
    # this function would allow it, and `rm -rf /` would be executed by the shell.
    # A safer alternative for executing commands constructed from parts is to use
    # `subprocess.run(parsed_cmd_parts, capture_output=True, text=True, check=False)`, 
    # which can avoid invoking a shell for interpreting arguments if `shell=False` (default for list args).
    
    with os.popen(shell_cmd) as pipe:
        output = pipe.read()
    
    # The `output` variable contains whatever the command wrote to its stdout and stderr.
    # os.popen does not easily provide the exit code of the command.
    return output
