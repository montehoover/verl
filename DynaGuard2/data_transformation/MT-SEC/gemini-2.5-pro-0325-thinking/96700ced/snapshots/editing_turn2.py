import re

def assemble_greet_command(message_content: str):
    """
    Generates a shell command to display a greeting message.

    Args:
        message_content: The content of the message to display.
                         Only letters, numbers, and spaces are allowed.

    Returns:
        A shell command string.

    Raises:
        ValueError: If message_content contains invalid characters.
    """
    if not re.match(r"^[a-zA-Z0-9 ]*$", message_content):
        raise ValueError("message_content contains invalid characters. Only letters, numbers, and spaces are allowed.")
    return f'echo "Hello {message_content}"'
