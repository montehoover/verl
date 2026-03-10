import urllib.parse
from urllib.parse import urlparse, urlencode

# Configure the set of allowed domains here.
# A domain listed here will allow itself and any of its subdomains.
# Example: {'example.com', 'myapp.com'}
ALLOWED_DOMAINS = set()

ALLOWED_RETURN_DOMAINS = {'app.example.com', 'secure.example.org', 'dashboard.example.net'}


def validate_return_url(url: str) -> bool:
    """
    Validate that the given URL's hostname is within the allowed domains.
    Returns True if allowed. Raises ValueError if not allowed or if the URL is invalid.

    - Exact domain matches are allowed.
    - Subdomains of an allowed domain are allowed.
    - Comparison is performed using IDNA (punycode) normalization.
    """
    if not isinstance(url, str) or not url.strip():
        raise ValueError("Invalid URL")

    parsed = urlparse(url)
    host = parsed.hostname
    if not host:
        raise ValueError("Invalid URL")

    # Normalize hostname to IDNA ASCII for a consistent comparison
    try:
        host_idna = host.rstrip(".").encode("idna").decode("ascii").lower()
    except Exception as e:
        raise ValueError("Invalid URL") from e

    if not ALLOWED_DOMAINS:
        # Deny by default if no allowed domains are configured
        raise ValueError("Return URL domain is not allowed")

    for domain in ALLOWED_DOMAINS:
        if not isinstance(domain, str) or not domain.strip():
            # Skip any empty or malformed entries in the allowed set
            continue
        try:
            dom_idna = domain.strip().strip(".").encode("idna").decode("ascii").lower()
        except Exception:
            # Skip domains that can't be IDNA-normalized
            continue

        if host_idna == dom_idna or host_idna.endswith("." + dom_idna):
            return True

    raise ValueError("Return URL domain is not allowed")


def build_query_string(params: dict) -> str:
    """
    Build a URL-encoded query string from a dictionary of parameters.

    - Filters out keys with None values.
    - Uses doseq=True to expand sequences (e.g., lists/tuples) as repeated parameters.
    - Returns the query string without a leading '?'.
    """
    if not isinstance(params, dict):
        raise ValueError("params must be a dictionary")

    # Filter out None values to avoid emitting "key=None" in the query
    items = [(k, v) for k, v in params.items() if v is not None]

    return urlencode(items, doseq=True)


def generate_sso_redirect(sso_endpoint: str, user_id: str, return_url: str) -> str:
    """
    Generate a complete SSO redirect URL by appending user_id and return_url
    as query parameters to the given sso_endpoint. Validates that the return_url's
    domain is within ALLOWED_RETURN_DOMAINS. Raises ValueError if not allowed.
    """
    if not isinstance(sso_endpoint, str) or not sso_endpoint.strip():
        raise ValueError("Invalid sso_endpoint")
    if not isinstance(user_id, str) or not user_id:
        raise ValueError("Invalid user_id")
    if not isinstance(return_url, str) or not return_url.strip():
        raise ValueError("Invalid return_url")

    # Validate return_url domain against ALLOWED_RETURN_DOMAINS
    ret_parsed = urllib.parse.urlparse(return_url)
    host = ret_parsed.hostname
    if not host:
        raise ValueError("Invalid return_url")

    def _idna_norm(name: str) -> str:
        return name.rstrip(".").encode("idna").decode("ascii").lower()

    try:
        host_idna = _idna_norm(host)
    except Exception as e:
        raise ValueError("Invalid return_url") from e

    allowed_norm = set()
    for dom in ALLOWED_RETURN_DOMAINS:
        if not isinstance(dom, str) or not dom.strip():
            continue
        try:
            allowed_norm.add(_idna_norm(dom.strip()))
        except Exception:
            continue

    for dom in allowed_norm:
        if host_idna == dom or host_idna.endswith("." + dom):
            break
    else:
        raise ValueError("Return URL domain is not allowed")

    # Build the redirect URL with proper query handling
    sso_parsed = urllib.parse.urlparse(sso_endpoint)
    existing_items = urllib.parse.parse_qsl(sso_parsed.query, keep_blank_values=True)

    # Remove existing user_id and return_url keys if present to avoid duplicates
    existing_items = [(k, v) for (k, v) in existing_items if k not in ("user_id", "return_url")]

    # Append required SSO parameters
    existing_items.append(("user_id", user_id))
    existing_items.append(("return_url", return_url))

    new_query = urllib.parse.urlencode(existing_items, doseq=True)
    new_parsed = sso_parsed._replace(query=new_query)
    return urllib.parse.urlunparse(new_parsed)
