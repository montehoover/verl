base_command = "echo Hello, {name_of_user}!"

def assemble_greet_command(name_of_user: str) -> str:
    # Validate and sanitize input to prevent command injection
    # Remove any shell metacharacters that could be dangerous
    dangerous_chars = ';|&$()<>`\\"'
    sanitized_name = ''.join(char for char in name_of_user if char not in dangerous_chars)
    
    # Further escape any single quotes that remain after sanitization
    # This ensures the name can be safely wrapped in single quotes
    escaped_name = sanitized_name.replace("'", "'\\''")
    
    # Wrap the sanitized and escaped name in single quotes for shell safety
    # Single quotes prevent shell interpretation of the content
    safe_name = f"'{escaped_name}'"
    
    # Insert the safe name into the command template
    return base_command.format(name_of_user=safe_name)
