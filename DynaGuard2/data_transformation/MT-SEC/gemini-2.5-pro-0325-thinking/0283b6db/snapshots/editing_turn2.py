from urllib.parse import urlparse, urlunparse, parse_qs, urlencode

def generate_user_redirect_url(base_url: str, user_id: str, return_url: str) -> str:
    """
    Generates a URL with user_id and return_url appended as query parameters.

    Args:
        base_url: The base URL (e.g., "http://example.com/profile").
        user_id: The user's ID.
        return_url: The URL to return to.

    Returns:
        A new URL string with user_id and return_url as query parameters
        (e.g., "http://example.com/profile?user_id=123&return_url=http%3A%2F%2Fprevious.com").
    """
    # Parse the base URL into its components
    parsed_url = urlparse(base_url)

    # Parse the existing query string into a dictionary.
    # parse_qs stores values as lists and preserves blank values if specified.
    query_dict = parse_qs(parsed_url.query, keep_blank_values=True)

    # Update the dictionary with the new user_id and return_url.
    # These will overwrite existing parameters with the same names.
    # Values are set as lists, consistent with parse_qs output.
    query_dict['user_id'] = [user_id]
    query_dict['return_url'] = [return_url]

    # Encode the updated dictionary back into a query string.
    # doseq=True handles list values correctly (e.g., param=value1&param=value2 if a list has multiple items,
    # or param=value if a list has a single item).
    new_query_string = urlencode(query_dict, doseq=True)

    # Reconstruct the URL with the new query string.
    # url_parts is a mutable sequence (list) derived from parsed_url's tuple structure:
    # (scheme, netloc, path, params, query, fragment)
    url_parts = list(parsed_url)
    url_parts[4] = new_query_string  # Index 4 corresponds to the 'query' component

    return urlunparse(url_parts)
