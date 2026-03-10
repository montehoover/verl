import urllib.parse

def generate_user_redirect_url(base_url: str, user_id: str, return_url: str) -> str:
    """
    Generate a URL that appends 'user_id' and 'return_url' as query parameters.

    Args:
        base_url: The base URL to which the parameters will be appended.
        user_id: The user identifier to add as a query parameter.
        return_url: The URL to return to, added as a query parameter.

    Returns:
        A URL string with the 'user_id' and 'return_url' query parameters appended.
    """
    parts = urllib.parse.urlparse(base_url)
    # Preserve existing query params and append new ones
    query_pairs = urllib.parse.parse_qsl(parts.query, keep_blank_values=True)
    query_pairs.append(('user_id', user_id))
    query_pairs.append(('return_url', return_url))

    new_query = urllib.parse.urlencode(query_pairs, doseq=True)
    new_parts = parts._replace(query=new_query)
    return urllib.parse.urlunparse(new_parts)
