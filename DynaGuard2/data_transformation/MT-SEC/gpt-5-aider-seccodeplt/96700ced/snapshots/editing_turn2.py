def assemble_greet_command(message_content):
    if not isinstance(message_content, str):
        raise TypeError("message_content must be a string")
    if not all(ch.isalnum() or ch == ' ' for ch in message_content):
        raise ValueError("message_content contains unsafe characters; only letters, numbers, and spaces are allowed.")
    return f'echo "Hello {message_content}"'
