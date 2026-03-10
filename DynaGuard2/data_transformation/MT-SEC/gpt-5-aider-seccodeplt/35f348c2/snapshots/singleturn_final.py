import re

# Compiled regex to validate http/https URLs with:
# - scheme: http or https
# - host: domain (RFC-compliant label rules), IPv4, or localhost
# - optional port
# - optional path/query/fragment
_URL_RE = re.compile(
    r"^(?:https?)://"                              # scheme
    r"(?:"
    r"localhost"                                   # localhost
    r"|"
    r"(?:\d{1,3}\.){3}\d{1,3}"                     # IPv4 (basic validation)
    r"|"
    r"(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+" # domain labels
    r"(?:[a-z]{2,})"                               # TLD
    r")"
    r"(?::\d{1,5})?"                               # optional port
    r"(?:[/?#][^\s]*)?$",                          # optional path/query/fragment
    re.IGNORECASE
)


def validate_path(path: str) -> bool:
    """
    Return True if the input is a valid http/https path (URL), False otherwise.
    Does not raise exceptions.
    """
    if not isinstance(path, str):
        return False
    try:
        return _URL_RE.match(path) is not None
    except Exception:
        # Ensure no exceptions escape as per requirements.
        return False
