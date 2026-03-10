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

def filter_and_execute_command(command: str, allowed_subcommands) -> str:
    """
    Execute a command only if its subcommands are allowed.
    Subcommands considered:
      - The base command (first token), e.g., 'git'
      - Optional second token if it is not an option (does not start with '-'),
        forming a two-part subcommand like 'git status'

    allowed_subcommands: iterable of strings (e.g., ['ls', 'git', 'git status'])
    """
    try:
        if not isinstance(command, str) or not command.strip():
            return "Error executing command: command must be a non-empty string"

        if not isinstance(allowed_subcommands, (list, tuple, set)):
            return "Error executing command: allowed_subcommands must be a list of strings"

        allowed = set(str(s) for s in allowed_subcommands)

        # Tokenize command without invoking the shell
        tokens = shlex.split(command, posix=(os.name != 'nt'))
        if not tokens:
            return "Error executing command: command must be a non-empty string"

        # Disallow common shell control/redirect operators even though shell=False
        forbidden_operators = {"|", "||", "&&", ";", ">", "<", ">>", "2>", "2>&1"}
        if any(tok in forbidden_operators for tok in tokens):
            return "Error executing command: command contains forbidden operators"

        # Derive subcommands to check
        base = tokens[0]
        subcommands_to_check = [base]
        if len(tokens) > 1 and not tokens[1].startswith("-"):
            subcommands_to_check.append(f"{base} {tokens[1]}")

        # Ensure all identified subcommands are allowed
        not_allowed = [sc for sc in subcommands_to_check if sc not in allowed]
        if not_allowed:
            return f"Error executing command: subcommand(s) not allowed: {', '.join(not_allowed)}"

        # Delegate execution to the safe executor
        return execute_command(command)

    except Exception as e:
        return f"Error executing command: {e}"
