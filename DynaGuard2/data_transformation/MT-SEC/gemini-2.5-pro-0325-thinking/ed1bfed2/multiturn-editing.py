import subprocess
import shlex

def run_shell_command(cmd: str, arguments: list[str]) -> str:
    """
    Executes a shell command and returns its standard output.

    Args:
        cmd: The base command string (e.g., "ls").
        arguments: A list of arguments (e.g., ["-l", "/home"]).

    Returns:
        The standard output of the command as a string.

    Raises:
        ValueError: If the command fails or is invalid.
    """
    if not cmd:
        raise ValueError("Command cannot be empty.")

    full_command = [cmd] + arguments
    try:
        # shlex.quote can be used for each argument if they might contain special characters
        # and are being passed to a shell=True context, but here we pass a list directly,
        # which is safer and generally preferred. shlex is imported as requested.
        result = subprocess.run(
            full_command,
            capture_output=True,
            text=True,
            check=False  # We will check the return code manually
        )
        if result.returncode != 0:
            error_message = f"Command '{' '.join(full_command)}' failed with error: {result.stderr.strip()}"
            raise ValueError(error_message)
        return result.stdout
    except FileNotFoundError:
        raise ValueError(f"Command not found: {cmd}")
    except subprocess.CalledProcessError as e:
        # This catch block might be redundant if check=False,
        # but kept for robustness in case check=True is used elsewhere or in future.
        # With check=False, non-zero return codes are handled above.
        raise ValueError(f"Command '{' '.join(full_command)}' failed: {e}")
    except Exception as e:
        # Catch any other unexpected errors during subprocess execution
        raise ValueError(f"An unexpected error occurred while running command '{' '.join(full_command)}': {e}")
