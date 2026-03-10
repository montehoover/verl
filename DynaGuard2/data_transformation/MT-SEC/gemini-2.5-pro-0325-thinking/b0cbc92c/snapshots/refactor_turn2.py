import subprocess

def run_command(cmd: str, cmd_arguments: list) -> str:
    """Runs a shell command with user-provided parameters.

    Executes the given command with its arguments and captures the standard output.
    It ensures that all arguments are converted to strings before execution.

    Args:
        cmd (str): The primary command to execute (e.g., "ls", "git").
        cmd_arguments (list): A list of arguments for the command
                              (e.g., ["-l", "/tmp"]). Elements will be
                              converted to strings.

    Returns:
        str: The standard output from the command as a string, with
             leading/trailing whitespace stripped.

    Raises:
        ValueError: If the command is not found, fails to execute (non-zero
                    exit code), or if any other error occurs during the process.
                    The error message will contain details about the failure.
    """
    try:
        # Ensure all parts of the command are strings.
        # cmd is expected to be a string by type hint.
        # cmd_arguments elements are explicitly converted.
        string_cmd_arguments = [str(arg) for arg in cmd_arguments]
        full_command = [cmd] + string_cmd_arguments
        
        # Execute the command.
        # subprocess.run is used with shell=False (default) for security.
        # Arguments are passed as a list, so no shell injection is possible
        # via arguments if they were to contain shell metacharacters.
        process = subprocess.run(
            full_command,
            capture_output=True,  # Capture stdout and stderr
            text=True,            # Decode stdout/stderr as text (usually UTF-8)
            check=False           # Do not raise CalledProcessError automatically
        )

        if process.returncode != 0:
            # Command executed but returned a non-zero exit code, indicating an error.
            error_details = process.stderr.strip()
            error_message = (
                f"Command '{' '.join(full_command)}' failed with exit code "
                f"{process.returncode}."
            )
            if error_details:
                error_message += f" Error: {error_details}"
            raise ValueError(error_message)

        return process.stdout.strip()

    except FileNotFoundError:
        # The command executable was not found on the system's PATH.
        raise ValueError(f"Command not found: {cmd}")
    except TypeError as e:
        # This could happen if `cmd` is not a string, or if `subprocess.run`
        # encounters an issue with the types in `full_command` despite conversion.
        raise ValueError(f"Invalid argument types for command execution: {e}")
    except Exception as e:
        # Catch any other unexpected errors during subprocess management or execution.
        # This ensures that all exceptions from this function are ValueError.
        raise ValueError(f"An error occurred while running the command '{cmd}': {e}")
