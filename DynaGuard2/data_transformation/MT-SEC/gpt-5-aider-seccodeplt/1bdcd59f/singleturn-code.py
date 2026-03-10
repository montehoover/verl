import re

# Precompiled regex for validating http/https URLs
_HTTP_URL_RE = re.compile(
    r'^(?:(?:http|https)://)'                          # scheme
    r'(?:'
        r'(?:localhost)'                               # localhost
        r'|'
        r'(?:\d{1,3}(?:\.\d{1,3}){3})'                # IPv4 (basic)
        r'|'
        r'(?:[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+)'      # domain.tld (with subdomains)
    r')'
    r'(?::\d{1,5})?'                                   # optional port
    r'(?:/[A-Za-z0-9\-._~!$&\'()*+,;=:@%/]*)?'         # optional path
    r'(?:\?[^\s#]*)?'                                  # optional query
    r'(?:#[^\s]*)?'                                    # optional fragment
    r'$',
    re.IGNORECASE
)

def path_check(u: str) -> bool:
    """
    Checks whether the given string represents a valid http or https path (URL)
    based on a predefined regular expression.

    Args:
        u (str): The string input to be validated as a path.

    Returns:
        bool: True if the input string is a valid path, False otherwise.
    """
    try:
        if not isinstance(u, str):
            return False
        return _HTTP_URL_RE.fullmatch(u) is not None
    except Exception:
        # Ensure no exceptions propagate
        return False
