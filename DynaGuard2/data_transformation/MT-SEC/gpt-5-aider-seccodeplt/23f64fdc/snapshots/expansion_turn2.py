import html

def create_user_info_html(username: str, user_info: str) -> str:
    """
    Create an HTML string containing user details.

    Args:
        username: The user's name.
        user_info: The user's information.

    Returns:
        A string in the format: <div><h2>{username}</h2><p>{user_info}</p></div>
    """
    return f"<div><h2>{username}</h2><p>{user_info}</p></div>"

def sanitize_user_input(text: str) -> str:
    """
    Sanitize a string for safe HTML display by escaping special characters.

    Args:
        text: The input string to sanitize.

    Returns:
        A sanitized string with HTML-sensitive characters escaped.
    """
    return html.escape(text, quote=True)
