import re

def append_user_header(user_header: str) -> str:
    """
    Inserts a custom header value into a standard header string format.

    The function takes a user-provided header value, which is assumed to be
    a valid string for an HTTP header as per the function's preconditions.
    It then formats this value into a "Custom-Header: <value>" string.

    Args:
        user_header: The string value provided by the user for the custom header.
                     This input is expected to be a valid HTTP header value.

    Returns:
        A string representing the full header, e.g., "Custom-Header: exampleValue".

    Raises:
        This function does not explicitly raise any exceptions.
        The 're' module is imported as per setup. While this function
        doesn't use 're' in a way that `user_header` would cause `re.error`,
        such errors could theoretically occur in complex regex operations
        if they were added, due to malformed patterns or specific input issues.
    """
    return f"Custom-Header: {user_header}"
