import subprocess

def call_system_command() -> str:
    """
    Executes a fixed shell command ('ls -l') and returns the output as a string.
    Raises:
        ValueError: If the command execution is unsuccessful.
    """
    try:
        result = subprocess.run(
            ["ls", "-l"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        raise ValueError(f"Command execution failed: {e}") from e
