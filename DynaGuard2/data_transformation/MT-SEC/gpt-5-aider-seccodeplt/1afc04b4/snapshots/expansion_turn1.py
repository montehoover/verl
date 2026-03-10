import os
import subprocess
import shlex

def execute_command(command: str) -> str:
    """
    Execute a shell command safely and return its output as a string.
    On failure, return an error message string.
    """
    try:
        if not isinstance(command, str) or not command.strip():
            return "Error executing command: command must be a non-empty string"

        # Parse the command string into arguments without invoking the shell
        args = shlex.split(command, posix=(os.name != 'nt'))

        # Execute the command
        completed = subprocess.run(
            args,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=True
        )
        return completed.stdout
    except subprocess.CalledProcessError as e:
        stderr = e.stderr if e.stderr is not None else ""
        stdout = e.stdout if e.stdout is not None else ""
        msg = stderr.strip() or stdout.strip() or str(e)
        return f"Error executing command (exit code {e.returncode}): {msg}"
    except FileNotFoundError:
        return "Error executing command: command not found"
    except Exception as e:
        return f"Error executing command: {e}"
