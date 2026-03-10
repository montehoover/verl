def assemble_greet_command(message_content):
    # Validate input contains only safe characters
    if not all(c.isalnum() or c.isspace() for c in message_content):
        raise ValueError("message_content must only contain letters, numbers, and spaces")
    
    return f'echo "Hello {message_content}"'
