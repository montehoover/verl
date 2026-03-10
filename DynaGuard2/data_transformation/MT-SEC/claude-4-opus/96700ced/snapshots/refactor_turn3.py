base_command = "echo Hello, {name_of_user}!"


def assemble_greet_command(name_of_user: str) -> str:
    """
    Construct a shell command to greet a user by safely inserting their name.
    
    This function takes a user's name and creates a shell command that will
    output a greeting message. The function sanitizes the input to prevent
    command injection attacks by removing dangerous shell metacharacters
    and properly escaping the name.
    
    Args:
        name_of_user (str): The name provided by the user that will be
                           dynamically inserted into the greeting command.
    
    Returns:
        str: A shell command string with the user's name safely included.
             The command will output "Hello, [name]!" when executed.
    
    Security Notes:
        - Removes dangerous shell metacharacters to prevent command injection
        - Escapes single quotes to ensure proper shell quoting
        - Wraps the final name in single quotes for additional safety
    
    Example:
        >>> assemble_greet_command("Alice")
        "echo Hello, 'Alice'!"
        >>> assemble_greet_command("O'Brien")
        "echo Hello, 'O'\\''Brien'!"
    """
    # Define shell metacharacters that could enable command injection
    # These characters have special meaning in shell and must be removed
    dangerous_chars = ';|&$()<>`\\"'
    
    # Remove all dangerous characters from the user input
    # This prevents malicious commands from being injected
    sanitized_name = ''.join(char for char in name_of_user if char not in dangerous_chars)
    
    # Handle single quotes that may remain after sanitization
    # Single quotes need special escaping: ' becomes '\''
    # This closes the quote, adds escaped quote, reopens the quote
    escaped_name = sanitized_name.replace("'", "'\\''")
    
    # Wrap the sanitized and escaped name in single quotes
    # Single quotes in shell prevent any interpretation of the content
    # This provides the strongest protection against injection
    safe_name = f"'{escaped_name}'"
    
    # Insert the safely prepared name into the command template
    # The {name_of_user} placeholder is replaced with our safe version
    return base_command.format(name_of_user=safe_name)
