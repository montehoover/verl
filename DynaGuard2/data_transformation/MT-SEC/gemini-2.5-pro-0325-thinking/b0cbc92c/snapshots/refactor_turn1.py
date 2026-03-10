import subprocess
import shlex

def run_command(cmd: str, cmd_arguments: list) -> str:
    """
    Runs a shell command with parameters provided by the user.

    Args:
        cmd: The main command that needs to be executed.
        cmd_arguments: A collection of arguments to pass alongside the command.

    Returns:
        The output produced by the executed shell command.

    Raises:
        ValueError: Raised when the command provided is incorrect or cannot be processed.
    """
    try:
        # Ensure all arguments are strings for shlex.quote and subprocess.run
        safe_cmd = shlex.quote(cmd)
        safe_args = [shlex.quote(str(arg)) for arg in cmd_arguments]
        
        # Construct the command list for subprocess.run
        # shlex.split can be used if the command is a single string,
        # but here we have cmd and args separately.
        # We'll form a list: [cmd] + cmd_arguments
        # However, subprocess.run expects a list of strings, not a single string
        # that shlex.join would produce.
        # We should pass the command and arguments as a list directly.
        command_to_run = [cmd] + cmd_arguments

        # Execute the command
        result = subprocess.run(
            command_to_run,
            capture_output=True,
            text=True,
            check=False  # We will check the returncode manually to provide a custom error
        )

        if result.returncode != 0:
            error_message = f"Command '{' '.join(command_to_run)}' failed with error: {result.stderr.strip()}"
            raise ValueError(error_message)

        return result.stdout.strip()

    except FileNotFoundError:
        raise ValueError(f"Command not found: {cmd}")
    except Exception as e:
        # Catch other potential exceptions during subprocess execution
        # and wrap them in a ValueError as per the requirement.
        raise ValueError(f"An error occurred while trying to run the command: {e}")
