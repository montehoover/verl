from urllib.parse import urlparse
import ipaddress

def extract_url_parts(url: str) -> dict:
    """
    Extract components of a URL string into a dictionary.

    Returns keys: scheme, netloc, path, params, query, fragment, username, password, hostname, port.
    """
    parsed = urlparse(url)
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
        "port": parsed.port,
    }

def is_valid_url(url: str) -> bool:
    """
    Validate a URL based on:
    - Scheme must be 'http' or 'https'
    - Hostname must not be localhost or a loopback IP (e.g., 127.0.0.1, ::1)

    Returns:
        True if valid, False otherwise.
    """
    try:
        parsed = urlparse(url)

        if parsed.scheme not in ("http", "https"):
            return False

        if not parsed.netloc:
            return False

        # Accessing parsed.port triggers validation of port number; ValueError if invalid.
        try:
            _ = parsed.port
        except ValueError:
            return False

        host = parsed.hostname
        if not host:
            return False

        host_l = host.lower()

        # Disallow localhost hostnames (including subdomains of .localhost)
        if host_l == "localhost" or host_l.endswith(".localhost"):
            return False

        # If it's an IP address, ensure it's not loopback.
        try:
            ip = ipaddress.ip_address(host_l)
            if ip.is_loopback:
                return False
        except ValueError:
            # Not an IP literal; allow non-localhost hostnames.
            pass

        return True
    except Exception:
        return False
