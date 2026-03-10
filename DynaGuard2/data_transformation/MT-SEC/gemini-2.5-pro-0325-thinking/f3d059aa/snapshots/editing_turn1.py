import re

def append_user_header(user_header: str) -> str:
    """
    Constructs a 'Content-Type' header string.

    Args:
        user_header: The MIME type provided by the user.

    Returns:
        A correctly formatted 'Content-Type' header string.
    """
    # For now, we'll assume the user_header is already a valid MIME type string.
    # No complex validation or formatting is requested beyond concatenation.
    return f"Content-Type: {user_header}"
