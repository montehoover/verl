import os
import shlex
import shutil
import subprocess
from typing import List, Tuple


def _split_command(shell_cmd: str) -> List[str]:
    try:
        # posix=True for consistent, predictable tokenization rules
        tokens = shlex.split(shell_cmd, posix=True)
    except ValueError as e:
        # Raised on unmatched quotes or bad escaping
        raise ValueError(f"Invalid command syntax: {e}") from e

    if not tokens:
        raise ValueError("shell_cmd must contain a command")

    return tokens


def _is_executable(cmd_name: str) -> bool:
    # If command includes a path component, verify it's executable
    if os.path.sep in cmd_name or (os.path.altsep and os.path.altsep in cmd_name):
        return os.path.isfile(cmd_name) and os.access(cmd_name, os.X_OK)

    # Otherwise, resolve from PATH
    return shutil.which(cmd_name) is not None


def _validate_tokens_structure(tokens: List[str]) -> None:
    # Basic structure: first token is a command name; subsequent tokens are options or args
    cmd = tokens[0]

    if not cmd or cmd.startswith("-"):
        raise ValueError("Invalid command: command name cannot be empty or start with '-'")

    if not _is_executable(cmd):
        raise ValueError(f"Command not found or not executable: '{cmd}'")

    # Disallow shell control/metacharacters to keep to a single simple command
    # This keeps validation simple and consistent with tokenized execution.
    prohibited_tokens = {"|", "||", "&", "&&", ";", "(", ")", "`"}
    for t in tokens:
        if t in prohibited_tokens:
            raise ValueError(f"Invalid command structure: unsupported shell operator '{t}'")

    # Validate option-like tokens until the first positional arg
    # We don't validate that options are correct for the specific command,
    # only that their shapes look like options.
    def is_option_like(tok: str) -> bool:
        if not tok.startswith("-"):
            return False
        # Accept:
        # - short options clusters like -la, -f, -abc
        # - long options like --all, --color=auto
        if tok.startswith("--"):
            # --opt or --opt=value
            if len(tok) <= 2:
                return False
            # Simple shape check: letters/digits/hyphens before optional =value
            if "=" in tok:
                name, value = tok.split("=", 1)
                if len(name) <= 2:
                    return False
                return name[2:].replace("-", "").isalnum()
            return tok[2:].replace("-", "").isalnum()
        else:
            # -a, -la, -abc, and allow -o=value
            if "=" in tok:
                name, value = tok.split("=", 1)
                if len(name) < 2:
                    return False
                return name[1:].isalnum()
            return len(tok) >= 2 and tok[1:].isalnum()

    # Once a non-option positional argument appears, allow remaining tokens freely (sans prohibited)
    positional_seen = False
    for tok in tokens[1:]:
        if positional_seen:
            continue
        if tok == "--":
            positional_seen = True
            continue
        if is_option_like(tok):
            continue
        positional_seen = True


def validate_and_execute_command(shell_cmd: str) -> str:
    """
    Execute the given shell command and return its stdout as a string.
    Raises ValueError on invalid input, structural validation failure, or any execution failure.
    """
    if not isinstance(shell_cmd, str):
        raise ValueError("shell_cmd must be a string")

    cmd = shell_cmd.strip()
    if not cmd:
        raise ValueError("shell_cmd must be a non-empty string")

    tokens = _split_command(cmd)
    _validate_tokens_structure(tokens)

    try:
        # Execute without invoking the shell to adhere to the validated token structure
        result = subprocess.run(
            tokens,
            shell=False,
            capture_output=True,
            text=True,
            env=os.environ,
        )
    except Exception as e:
        raise ValueError(f"Failed to execute command: {e}") from e

    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        stdout = (result.stdout or "").strip()
        err_msg = f"Command failed with exit code {result.returncode}"
        if stderr:
            err_msg += f": {stderr}"
        elif stdout:
            err_msg += f": {stdout}"
        raise ValueError(err_msg)

    return result.stdout
