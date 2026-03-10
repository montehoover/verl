import shlex


# Template for the shell command with placeholder for recipient name
base_command = "echo Hello, {recipient}!"


def compose_hello_command(recipient: str) -> str:
    """
    Construct a shell command to greet a user by name.
    
    This function takes a recipient name and safely inserts it into a 
    predefined shell command template. The name is properly escaped to 
    prevent shell injection vulnerabilities.
    
    Args:
        recipient: The name of the person to greet. This will be safely
                  escaped before insertion into the command.
    
    Returns:
        A string containing the complete shell command with the recipient's
        name safely included.
    
    Example:
        >>> compose_hello_command("Alice")
        "echo Hello, Alice!"
        >>> compose_hello_command("Alice'; rm -rf /")
        "echo Hello, 'Alice'\"'\"'; rm -rf /'!"
    """
    # Escape the recipient name to prevent shell injection attacks
    safe_recipient = shlex.quote(recipient)
    
    # Format the command template with the safely escaped recipient name
    formatted_command = base_command.format(recipient=safe_recipient)
    
    return formatted_command
