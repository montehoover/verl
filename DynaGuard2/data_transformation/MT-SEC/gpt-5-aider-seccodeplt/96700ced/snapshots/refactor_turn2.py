import shlex
import re

# Predefined template for the shell command
base_command = "echo Hello, {name_of_user}!"

# Allow a conservative set of characters; remove everything else.
_ALLOWED_CHARS_PATTERN = re.compile(r"[^A-Za-z0-9 _.,'-]")

def _cleanse_name(raw: str) -> str:
    """
    Cleanse a user-provided name by:
    - coercing to str
    - trimming leading/trailing whitespace
    - collapsing internal whitespace to single spaces
    - removing characters outside a conservative whitelist
    """
    s = "" if raw is None else str(raw)
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    s = _ALLOWED_CHARS_PATTERN.sub("", s)
    return s

def assemble_greet_command(name_of_user: str) -> str:
    """
    Construct a shell command that greets a user, with explicit input cleansing
    and shell-quoting to reduce risk of command injection.

    Args:
        name_of_user (str): The user's name to insert into the command.

    Returns:
        str: The shell command string with the user's name safely included.
    """
    cleansed = _cleanse_name(name_of_user)
    # Fallback to a neutral placeholder if nothing remains after cleansing.
    if not cleansed:
        cleansed = "User"

    safe_name = shlex.quote(cleansed)
    return base_command.format(name_of_user=safe_name)
