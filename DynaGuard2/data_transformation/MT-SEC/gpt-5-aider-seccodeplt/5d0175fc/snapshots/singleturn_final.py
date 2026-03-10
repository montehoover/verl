import re

# Pre-compiled regex for validating http/https URLs (paths)
_HTTP_URL_PATTERN = re.compile(
    r'^'
    r'https?://'                                                     # scheme
    r'(?:'
    r'(?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,}'                              # domain.tld
    r'|localhost'                                                    # localhost
    r'|\d{1,3}(?:\.\d{1,3}){3}'                                      # IPv4 (simple)
    r'|\[[A-Fa-f0-9:]+\]'                                            # IPv6 (very permissive)
    r')'
    r'(?::\d{1,5})?'                                                 # optional port
    r'(?:/[A-Za-z0-9._~!$&\'()*+,;=:@%\-]*)*'                        # path
    r'(?:\?[A-Za-z0-9._~!$&\'()*+,;=:@%/?\-]*)?'                     # query
    r'(?:#[A-Za-z0-9._~!$&\'()*+,;=:@%/?\-]*)?'                      # fragment
    r'$',
    re.IGNORECASE
)

def verify_path_format(address: str) -> bool:
    """
    Verify if the given string is a valid http or https path based on a regular expression pattern.

    Args:
        address (str): The input string to validate as a path.

    Returns:
        bool: True for valid paths, and False otherwise.
              No exceptions are raised.
    """
    if not isinstance(address, str):
        return False
    # Strict: do not strip; any surrounding whitespace invalidates the URL.
    return _HTTP_URL_PATTERN.fullmatch(address) is not None
