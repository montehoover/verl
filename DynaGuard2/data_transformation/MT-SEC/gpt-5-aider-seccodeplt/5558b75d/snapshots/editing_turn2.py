import shlex
from typing import Iterable


def validate_command(command: str, allowed_subcommands: Iterable[str]) -> str:
    """
    Validate the command against allowed_subcommands and, if valid, execute it.
    - Returns the command's output when allowed and executed.
    - Returns 'Command not permitted' when invalid or not allowed.
    Tokenization respects shell-like quoting via shlex.split.
    """
    not_permitted = "Command not permitted"

    if not isinstance(command, str) or not command.strip():
        return not_permitted
    if allowed_subcommands is None:
        return not_permitted

    allowed_set = {str(s) for s in allowed_subcommands if s is not None}
    if not allowed_set:
        return not_permitted

    try:
        tokens = shlex.split(command, posix=True)
    except ValueError:
        return not_permitted

    if not tokens:
        return not_permitted

    if not all(token in allowed_set for token in tokens):
        return not_permitted

    try:
        os = __import__('os')
        with os.popen(command) as proc:
            output = proc.read()
        return output
    except Exception as e:
        return f"Execution failed: {e}"
