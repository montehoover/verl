from typing import Any, Dict, Optional
from urllib.parse import urlparse, urlencode

def parse_and_validate_url(url: str) -> Dict[str, Any]:
    """
    Parse and validate a URL string.

    Returns a dictionary of parsed components:
    - scheme
    - netloc
    - path
    - params
    - query
    - fragment
    - username
    - password
    - hostname
    - port

    Raises:
        ValueError: If the URL is not a string, empty, contains whitespace,
                    or is structurally invalid for its scheme.
    """
    if not isinstance(url, str):
        raise ValueError("URL must be a string")

    candidate = url.strip()
    if not candidate:
        raise ValueError("URL is empty")

    # Reject unencoded whitespace inside the URL
    if any(ch.isspace() for ch in candidate):
        raise ValueError("URL must not contain unencoded whitespace")

    parsed = urlparse(candidate)

    if not parsed.scheme:
        raise ValueError("URL is missing a scheme (e.g., 'http', 'https')")

    # Determine validity based on the kind of scheme.
    network_schemes = {"http", "https", "ftp", "ftps", "ws", "wss"}
    if parsed.scheme in network_schemes:
        # Network URLs must have a network location (authority/host)
        if not parsed.netloc:
            raise ValueError("Network URL must include a host (netloc)")
        if parsed.hostname is None:
            raise ValueError("URL host is invalid or missing")
        # Validate port if present
        try:
            _ = parsed.port  # Accessing .port validates it
        except ValueError:
            raise ValueError("Port is invalid")
    elif parsed.scheme == "file":
        # file URLs require a path; netloc may be empty or a host
        if not parsed.path:
            raise ValueError("file URL must include a path")
    else:
        # Generic validation for other schemes (mailto, data, etc.)
        if not (parsed.netloc or parsed.path):
            raise ValueError("URL must include either a host (netloc) or a path")

    # Return all commonly useful components
    try:
        port: Optional[int] = parsed.port
    except ValueError:
        # Already validated above for network schemes, but guard anyway
        raise ValueError("Port is invalid")

    return {
        "scheme": parsed.scheme,
        "netloc": parsed.netloc,
        "path": parsed.path,
        "params": parsed.params,
        "query": parsed.query,
        "fragment": parsed.fragment,
        "username": parsed.username,
        "password": parsed.password,
        "hostname": parsed.hostname,
        "port": port,
    }


def build_query_string(params: Dict[str, Any]) -> str:
    """
    Build a URL-encoded query string from a dictionary of parameters.

    Rules:
    - Keys are converted to strings.
    - Values:
        - None values are omitted.
        - bool values become "true"/"false".
        - int/float/other values are stringified.
        - list/tuple values produce repeated keys (doseq behavior), with
          None elements omitted and other elements normalized as above.
    - Returns the query string without a leading '?'.
    """
    if not isinstance(params, dict):
        raise ValueError("params must be a dictionary")

    def normalize_scalar(value: Any) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)):
            return str(value)
        if isinstance(value, str):
            return value
        return str(value)

    # Prepare items suitable for urlencode with doseq=True
    prepared: list[tuple[str, Any]] = []
    for key, value in params.items():
        if value is None:
            continue
        k = str(key)
        if isinstance(value, (list, tuple)):
            seq = []
            for v in value:
                if v is None:
                    continue
                seq.append(normalize_scalar(v))
            if seq:
                prepared.append((k, seq))
        else:
            prepared.append((k, normalize_scalar(value)))

    if not prepared:
        return ""

    return urlencode(prepared, doseq=True)
