"""
Utilities for safely constructing a shell command that greets a user.

This module defines a small pipeline that:
- Cleanses a user-provided name to remove potentially dangerous characters.
- Applies shell quoting to the cleansed name.
- Injects the safely prepared value into a predefined shell command template.

Note:
    Improper modification of the cleansing or quoting steps can introduce
    security vulnerabilities such as command injection. The functions below
    attempt to reduce that risk by combining explicit input cleansing with
    POSIX-compliant shell quoting.
"""

import re
import shlex

# Predefined template for the shell command. The placeholder is substituted
# with a safely handled user name.
base_command = "echo Hello, {name_of_user}!"

# Allow a conservative set of characters; remove everything else.
# This whitelist includes common characters found in names and simple punctuation.
_ALLOWED_CHARS_PATTERN = re.compile(r"[^A-Za-z0-9 _.,'-]")


def _cleanse_name(raw: str) -> str:
    """
    Cleanse a user-provided name to reduce the risk of command injection.

    The cleansing process:
        1. Coerces the input to a string (None becomes an empty string).
        2. Trims leading and trailing whitespace.
        3. Collapses internal whitespace sequences into a single space.
        4. Removes any characters not in a conservative whitelist of
           alphanumerics, space, and a few punctuation marks: " _.,'- ".

    Args:
        raw (str): The raw user input to cleanse. May be None or any object
            convertible to string.

    Returns:
        str: A sanitized version of the user name containing only allowed
        characters. The result may be an empty string if all characters
        were removed.
    """
    # Normalize None to an empty string; coerce all other values to str.
    s = "" if raw is None else str(raw)

    # Trim surrounding whitespace.
    s = s.strip()

    # Replace any run of whitespace with a single space for readability.
    s = re.sub(r"\s+", " ", s)

    # Remove characters not matching the allowed set (whitelist-based).
    s = _ALLOWED_CHARS_PATTERN.sub("", s)

    return s


def assemble_greet_command(name_of_user: str) -> str:
    """
    Construct a shell command that greets a user, with explicit input cleansing
    and shell quoting to mitigate command injection risks.

    The function:
        - Cleanses the provided name using a conservative whitelist.
        - Falls back to the neutral placeholder "User" if the cleansed name
          becomes empty.
        - Applies shlex.quote to ensure the value is safely represented as a
          single shell argument.
        - Inserts the safely prepared value into the global base_command
          template.

    Args:
        name_of_user (str): The user's name to insert into the command.

    Returns:
        str: A shell command string (e.g., "echo Hello, 'Alice'!") with the
        user's name safely included.

    Note:
        This function does not raise specific exceptions for invalid input.
        However, improper changes to its cleansing or quoting steps could
        introduce security vulnerabilities.
    """
    # Cleanse the input to remove potentially unsafe characters.
    cleansed = _cleanse_name(name_of_user)

    # Fallback to a neutral placeholder if the name becomes empty after cleansing.
    if not cleansed:
        cleansed = "User"

    # Shell-quote the cleansed name so it is treated as a single argument.
    safe_name = shlex.quote(cleansed)

    # Interpolate the safely prepared name into the command template.
    return base_command.format(name_of_user=safe_name)
