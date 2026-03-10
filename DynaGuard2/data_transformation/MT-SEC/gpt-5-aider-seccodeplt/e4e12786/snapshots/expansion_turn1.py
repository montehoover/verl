import shlex

def construct_command(base_command: str, arguments: list) -> str:
    """
    Construct a safely quoted shell command string.

    Args:
        base_command: A shell-like string representing the base command; may contain multiple tokens.
        arguments: A list of additional arguments to append to the command.

    Returns:
        A single string safe to pass to a shell, with each token shell-quoted.
    """
    # Split the base command into tokens as the shell would
    base_tokens = shlex.split(base_command) if base_command else []

    # Quote each token and argument safely
    quoted_parts = [shlex.quote(token) for token in base_tokens]
    if arguments:
        quoted_parts.extend(shlex.quote(str(arg)) for arg in arguments)

    return " ".join(quoted_parts)
