import subprocess
import shlex

def run_shell_command(cmd: str, arguments: list) -> str:
    """
    Runs a shell command with user-specified arguments.

    Args:
        cmd: The primary command to run.
        arguments: A list containing arguments for the command.

    Returns:
        The result output (stdout) from the executed command.

    Raises:
        ValueError: Raised in case the command is invalid or fails to execute.
    """
    command_list = [cmd] + arguments
    try:
        # Ensure arguments are strings, as subprocess.run expects them.
        command_list_str = [str(arg) for arg in command_list]
        
        # shlex.join can be useful for logging or debugging the command string
        # full_command_str = shlex.join(command_list_str)
        # print(f"Executing: {full_command_str}")

        result = subprocess.run(
            command_list_str,
            capture_output=True,
            text=True,
            check=True  # This will raise CalledProcessError for non-zero exit codes
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        error_message = f"Command '{shlex.join(command_list_str)}' failed with exit code {e.returncode}.\nStderr: {e.stderr}"
        raise ValueError(error_message) from e
    except FileNotFoundError:
        raise ValueError(f"Command not found: {cmd}") from None
    except Exception as e:
        # Catch any other unexpected errors during subprocess execution
        raise ValueError(f"An unexpected error occurred while trying to run command '{shlex.join(command_list_str)}': {e}") from e
