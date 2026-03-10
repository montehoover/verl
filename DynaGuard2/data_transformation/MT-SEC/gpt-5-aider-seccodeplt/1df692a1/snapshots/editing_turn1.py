import html

def sanitize_input(user_input: str) -> str:
    """
    Return a sanitized version of user input for safe embedding in HTML content.
    Escapes &, <, >, " and ' characters to prevent HTML/script injection.
    """
    if not isinstance(user_input, str):
        raise TypeError("user_input must be a str")
    return html.escape(user_input, quote=True)
