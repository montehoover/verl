import subprocess

def call_system_command(options=None) -> str:
    """
    Executes the fixed shell command ('ls') with optional options and returns the output as a string.
    Args:
        options: An iterable of option strings (e.g., ["-l", "-a"]). Defaults to no options.
    Raises:
        ValueError: If options are invalid or the command execution is unsuccessful.
    """
    if options is None:
        opts = []
    else:
        try:
            opts = list(options)
        except TypeError as e:
            raise ValueError("options must be an iterable of strings") from e
        if not all(isinstance(o, str) for o in opts):
            raise ValueError("all options must be strings")

    try:
        result = subprocess.run(
            ["ls", *opts],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        err = getattr(e, "stderr", None)
        msg = err if err else str(e)
        raise ValueError(f"Command execution failed: {msg}") from e
