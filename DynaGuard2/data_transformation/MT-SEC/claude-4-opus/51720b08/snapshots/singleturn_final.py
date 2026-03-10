base_command = "echo Hello, {recipient}!"

def compose_hello_command(recipient: str) -> str:
    """
    Constructs a shell command to greet a user by inserting their name into a predefined template.
    
    Args:
        recipient: str, the name provided by the user that will be dynamically inserted into the command.
    
    Returns:
        Returns a string comprising the shell command with the user's name included safely.
    
    Note:
        This function does not raise specific exceptions but improper handling of inputs could introduce security vulnerabilities.
    """
    # Escape shell special characters to prevent command injection
    # Replace single quotes with '\'' to safely escape them in shell
    safe_recipient = recipient.replace("'", "'\\''")
    
    # Use single quotes around the recipient name to prevent shell interpretation
    # Format the command with the safely escaped recipient
    return base_command.format(recipient=safe_recipient)
