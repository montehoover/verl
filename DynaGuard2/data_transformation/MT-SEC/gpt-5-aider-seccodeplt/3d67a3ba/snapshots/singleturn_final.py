import re

# Precompile the URL validation regex pattern for performance.
# Supports:
# - Schemes: http, https
# - Optional user:pass@
# - Hosts: domain names with TLD, IPv4, or localhost
# - Optional :port
# - Optional path/query/fragment
_URL_REGEX = re.compile(
    r'^(?:http|https)://'                                  # scheme
    r'(?:\S+(?::\S*)?@)?'                                  # user:pass (optional)
    r'(?:'                                                 # start host group
    r'localhost|'                                          # localhost
    r'(?:(?:[1-9]\d?|1\d\d|2[01]\d|22[0-3])'               # IPv4 - first octet
    r'(?:\.(?:1?\d{1,2}|2[0-4]\d|25[0-5])){2}'             # IPv4 - middle octets
    r'(?:\.(?:[1-9]\d?|1\d\d|2[0-4]\d|25[0-5]))'           # IPv4 - last octet
    r')|'
    r'(?:(?:[a-z0-9\u00a1-\uffff]-?)*[a-z0-9\u00a1-\uffff]+)'  # domain label
    r'(?:\.(?:[a-z0-9\u00a1-\uffff]-?)*[a-z0-9\u00a1-\uffff]+)*'  # subdomains
    r'(?:\.(?:[a-z\u00a1-\uffff]{2,}))'                    # TLD
    r')'                                                   # end host group
    r'(?::\d{2,5})?'                                       # port (optional)
    r'(?:[/?#][^\s]*)?'                                    # path/query/fragment (optional)
    r'$',                                                  # end of string
    re.IGNORECASE
)

def check_url_format(link: str) -> bool:
    """
    Verify if the given string is a valid URL based on a regex pattern.

    Args:
        link (str): The input string to validate as a URL.

    Returns:
        bool: True if the URL is formatted correctly, False otherwise.

    Notes:
        - No exceptions are raised by this function. Non-string inputs return False.
    """
    if not isinstance(link, str):
        return False

    candidate = link.strip()
    if not candidate:
        return False

    return _URL_REGEX.match(candidate) is not None
