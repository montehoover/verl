import subprocess
import shlex
import os

def execute_shell_command(directory: str):
    """
    Lists all files in a directory specified by the user.

    Args:
        directory: The path to the directory.

    Prints:
        A single string with filenames separated by commas.

    Raises:
        ValueError: If the directory does not exist or the command fails.
    """
    if not os.path.isdir(directory):
        raise ValueError(f"Directory does not exist: {directory}")

    command = f"ls {shlex.quote(directory)}"
    try:
        process = subprocess.run(
            command,
            shell=True,  # Using shell=True for simplicity with ls, but be cautious
            check=True,    # Raises CalledProcessError if return code is non-zero
            capture_output=True,
            text=True
        )
        # process.stdout will contain filenames separated by newlines
        filenames = process.stdout.strip().split('\n')
        print(",".join(filenames))
    except subprocess.CalledProcessError as e:
        raise ValueError(f"Command failed: {e}")
    except FileNotFoundError: # Should not happen with shell=True and ls, but good practice
        raise ValueError(f"Command 'ls' not found. Ensure it is in your PATH.")
    except Exception as e: # Catch any other unexpected errors
        raise ValueError(f"An unexpected error occurred: {e}")
