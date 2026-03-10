import urllib.parse

def generate_user_url(base_url: str, user_id: str) -> str:
    """
    Generate a URL that appends the user_id as a query parameter.

    Args:
        base_url: The base URL to which the user_id parameter will be appended.
        user_id: The user identifier to add as a query parameter.

    Returns:
        A URL string with the user_id query parameter appended.
    """
    if '?' in base_url:
        if base_url.endswith('?') or base_url.endswith('&'):
            sep = ''
        else:
            sep = '&'
    else:
        sep = '?'
    return f"{base_url}{sep}user_id={urllib.parse.quote_plus(user_id)}"
