import shlex
from typing import Iterable


def validate_command(command: str, allowed_subcommands: Iterable[str]) -> bool:
    """
    Return True if all tokens in the command are within allowed_subcommands, else False.
    Tokenization respects shell-like quoting via shlex.split.
    """
    if not isinstance(command, str) or not command.strip():
        return False
    if allowed_subcommands is None:
        return False

    allowed_set = {str(s) for s in allowed_subcommands if s is not None}
    if not allowed_set:
        return False

    try:
        tokens = shlex.split(command, posix=True)
    except ValueError:
        return False

    if not tokens:
        return False

    return all(token in allowed_set for token in tokens)
