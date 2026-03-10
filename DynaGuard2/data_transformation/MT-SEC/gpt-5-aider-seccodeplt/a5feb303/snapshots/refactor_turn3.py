from urllib.parse import urlparse, urlunparse
import logging

ALLOWED_REDIRECT_DOMAINS = ["trusted-redirect.com", "partner.com"]

# Configure logging to a human-readable file in the current working directory
logger = logging.getLogger("redirects")
logger.setLevel(logging.INFO)
if not logger.handlers:
    file_handler = logging.FileHandler("redirects.log", encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


def is_domain_allowed(host: str, allowed_domains: list[str]) -> bool:
    """
    Pure function to check if a given host is allowed based on the allowed domains list.
    Allows exact domain matches or subdomains of any allowed domain.
    """
    if not host:
        return False
    host_l = host.lower().rstrip(".")
    for d in allowed_domains:
        dl = d.lower().rstrip(".")
        if host_l == dl or host_l.endswith("." + dl):
            return True
    return False


def _strip_ctrl(s: str) -> str:
    """Remove ASCII control characters (including DEL) from a string."""
    return "".join(ch for ch in s if ch >= " " and ch != "\x7f")


def sanitize_redirect_url(parsed, scheme: str) -> str:
    """
    Pure function that takes a parsed URL and a normalized scheme (http/https),
    validates structural components (port range, hostname encoding), strips unsafe
    characters, removes userinfo and fragments, and returns a sanitized URL string.
    """
    # Ensure a hostname exists
    hostname = parsed.hostname
    if not hostname:
        raise ValueError("Invalid URL: missing hostname")

    host_l = hostname.lower().rstrip(".")

    # Validate port if present
    port = parsed.port
    if port is not None:
        if port <= 0 or port > 65535:
            raise ValueError("Invalid URL: invalid port")

    # IDNA-encode the host to ASCII
    try:
        safe_host = host_l.encode("idna").decode("ascii")
    except Exception:
        raise ValueError("Invalid URL: invalid hostname")

    # Build sanitized netloc (exclude any userinfo)
    if port is not None:
        default_port = (scheme == "http" and port == 80) or (scheme == "https" and port == 443)
        netloc = safe_host if default_port else f"{safe_host}:{port}"
    else:
        netloc = safe_host

    # Sanitize path, params, query; ensure path starts with "/"
    path = _strip_ctrl(parsed.path or "/")
    if not path.startswith("/"):
        path = "/" + path

    params = _strip_ctrl(parsed.params or "")
    query = _strip_ctrl(parsed.query or "")

    # Drop fragment entirely
    fragment = ""

    return urlunparse((scheme, netloc, path, params, query, fragment))


def process_user_redirect(input_url: str) -> str:
    """
    Accepts a user-provided URL and returns a sanitized URL safe for redirection.
    Raises ValueError if the domain is not in ALLOWED_REDIRECT_DOMAINS or if the URL is invalid.
    Also logs each redirection attempt and its outcome to a file in the current working directory.
    """
    logger.info("Redirect attempt started. input_url='%s'", input_url)

    if not isinstance(input_url, str) or not input_url.strip():
        logger.info(
            "Redirect denied. Reason='empty or non-string input'. allowed=False. sanitized_url=None"
        )
        raise ValueError("Invalid URL: empty input")

    raw = input_url.strip()

    # Parse the URL
    parsed = urlparse(raw)

    # Handle scheme-relative URLs (e.g., //example.com/path) by defaulting to https
    if not parsed.scheme and parsed.netloc:
        scheme = "https"
    else:
        scheme = (parsed.scheme or "").lower()

    # Only allow HTTP(S) redirects
    if scheme not in ("http", "https"):
        logger.info(
            "Redirect denied. input_url='%s' Reason='unsupported scheme: %s'. allowed=False. sanitized_url=None",
            input_url,
            scheme or "(none)",
        )
        raise ValueError("Invalid URL: unsupported scheme")

    # Ensure a hostname exists for domain validation
    hostname = parsed.hostname
    if not hostname:
        logger.info(
            "Redirect denied. input_url='%s' Reason='missing hostname'. allowed=False. sanitized_url=None",
            input_url,
        )
        raise ValueError("Invalid URL: missing hostname")

    host_l = hostname.lower().rstrip(".")

    # Check against allowed domains (exact match or subdomain)
    if not is_domain_allowed(host_l, ALLOWED_REDIRECT_DOMAINS):
        logger.info(
            "Redirect denied. input_url='%s' Reason='domain not allowed'. domain='%s'. allowed=False. sanitized_url=None",
            input_url,
            host_l,
        )
        raise ValueError("Invalid URL: domain not allowed")

    # Sanitize and return the final redirect URL
    try:
        sanitized = sanitize_redirect_url(parsed, scheme)
    except ValueError as e:
        logger.info(
            "Redirect denied after sanitization checks. input_url='%s' Reason='%s'. allowed=False. sanitized_url=None",
            input_url,
            str(e),
        )
        raise

    logger.info(
        "Redirect allowed. input_url='%s' domain='%s' scheme='%s' allowed=True sanitized_url='%s'",
        input_url,
        host_l,
        scheme,
        sanitized,
    )
    return sanitized
