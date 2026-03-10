import re

def check_url_format(link: str) -> bool:
    """
    Verifies if a given string is a valid URL based on a regular expression pattern.

    The function uses a comprehensive regex to check for common URL structures,
    including the scheme (http, https, ftp), domain name (or localhost or IP address),
    an optional port, and an optional path/query/fragment.

    Args:
        link (str): The input string to validate as a URL.

    Returns:
        bool: True if the URL is formatted correctly according to the regex, False otherwise.
              No exceptions are raised by this function.
    """
    # Regular expression for URL validation.
    # This pattern is designed to be a general-purpose URL validator.
    # It can be broken down as follows:
    # - `^`: Asserts position at the start of the string.
    # - `(?:http|ftp)s?://`: Matches the scheme (http, https, ftp, ftps) followed by '://'.
    #   - `(?:...)`: Non-capturing group.
    #   - `s?`: Optionally matches 's'.
    # - `(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|localhost|\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})`: Matches the domain part.
    #   - `(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+`: Matches subdomains and the main domain name (e.g., "www.example.").
    #     - `[A-Z0-9]`: Matches an alphanumeric character.
    #     - `(?:[A-Z0-9-]{0,61}[A-Z0-9])?`: Optionally matches a sequence of alphanumeric characters or hyphens, up to 61 characters, ending with an alphanumeric character.
    #     - `\.`: Matches a literal dot.
    #   - `(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)`: Matches the top-level domain (TLD) (e.g., "com", "org") or a country code TLD.
    #   - `|localhost`: Alternatively, matches "localhost".
    #   - `|\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}`: Alternatively, matches an IP address (e.g., "192.168.1.1").
    # - `(?::\d+)?`: Optionally matches a port number (e.g., ":8080").
    #   - `:\d+`: Matches a colon followed by one or more digits.
    # - `(?:/?|[/?]\S+)`: Matches an optional path, query parameters, or fragment identifier.
    #   - `/?`: Optionally matches a single slash.
    #   - `|[/?]\S+`: Alternatively, matches a slash or question mark followed by one or more non-whitespace characters.
    # - `$`: Asserts position at the end of the string.
    # - `re.IGNORECASE`: Makes the matching case-insensitive.
    
    url_pattern = re.compile(
        r'^(?:http|ftp)s?://'  
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'
        r'localhost|'
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'
        r'(?::\d+)?'
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    
    # Perform a full match of the pattern against the input link.
    # Using bool() to convert the match object (or None) to True/False.
    return bool(re.fullmatch(url_pattern, link))
