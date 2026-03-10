import html

def sanitize_input(user_input: str) -> str:
    """Sanitize user input to prevent HTML/script injection attacks."""
    return html.escape(user_input)
