def sanitize_input(user_input: str) -> str:
    """
    Return a POSIX shell-safe representation of user_input.

    The value is single-quoted, and any embedded single quotes are safely
    handled by closing, inserting an escaped quote, and reopening:
    "abc" -> 'abc'
    "a'b" -> 'a'"'"'b'

    This helps prevent command injection when interpolating user-controlled
    values into shell command strings.
    """
    if not isinstance(user_input, str):
        raise TypeError("user_input must be a str")

    # Remove NUL bytes which cannot be represented in argv on most systems
    user_input = user_input.replace("\x00", "")

    if user_input == "":
        return "''"

    # Safe single-quoting per POSIX shells
    return "'" + user_input.replace("'", "'\"'\"'") + "'"


def construct_shell_string(safe_input: str) -> str:
    """
    Construct a shell-compatible command string that incorporates the provided
    safe_input, which must already be properly shell-quoted (e.g., produced by
    sanitize_input). Example output: "echo 'value'".
    """
    if not isinstance(safe_input, str):
        raise TypeError("safe_input must be a str")

    # Do not modify or re-quote; assume safe_input is already shell-safe.
    # Build a simple shell command structure that uses the safe argument.
    return f"echo {safe_input}"
