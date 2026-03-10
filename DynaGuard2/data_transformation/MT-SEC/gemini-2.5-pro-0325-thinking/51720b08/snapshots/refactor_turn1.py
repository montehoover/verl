import shlex

base_command = "echo Hello, {recipient}!"

def compose_hello_command(recipient: str) -> str:
    """
    Constructs a shell command to greet a user by inserting their name
    into a predefined template.

    Args:
        recipient: The name provided by the user that will be dynamically
                   inserted into the command.

    Returns:
        A string comprising the shell command with the user's name included safely.

    Raises:
        This function does not raise specific exceptions but improper handling
        of inputs could introduce security vulnerabilities if not handled carefully
        (e.g., by using shlex.quote).
    """
    # Sanitize the recipient's name to prevent command injection
    safe_recipient = shlex.quote(recipient)
    
    # Construct the command using the sanitized name
    # Note: The base_command itself uses a placeholder that looks like a format string.
    # We need to be careful if the base_command could come from an untrusted source.
    # Here, base_command is predefined, so direct formatting is acceptable
    # after the recipient part is sanitized.
    # However, the placeholder in base_command is {recipient}, not {safe_recipient}.
    # A more robust way if base_command was dynamic would be to replace a known placeholder.
    # For this specific problem, we format the string.
    
    # If the intention is to replace "{recipient}" in the string `base_command`
    # with the *quoted* version of the recipient, then we should do:
    # command = base_command.replace("{recipient}", safe_recipient)
    # This is generally safer than using .format() if the template string
    # might contain other curly braces that are not intended as format specifiers.

    # Given the setup `base_command = "echo Hello, {recipient}!"`
    # and the requirement to insert the name, using .format() with a
    # dictionary or keyword arguments is a common way.
    # The key here is that `safe_recipient` is already quoted.
    
    # The placeholder is {recipient}, so we should format with `recipient=safe_recipient`
    command = base_command.format(recipient=safe_recipient)
    
    return command
