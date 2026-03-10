import logging
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode, quote

AUTHORIZED_REDIRECT_DOMAINS = ["trusted-redirect.com", "partner.com"]

logger = logging.getLogger(__name__)


def normalize_hostname_idna(hostname: str) -> str:
    """
    Normalize a hostname to IDNA ASCII form, lowercase, and without a trailing dot.
    Raises ValueError if normalization fails or hostname is empty.
    """
    if not hostname:
        raise ValueError("URL must include a valid host")
    try:
        return hostname.encode("idna").decode("ascii").rstrip(".").lower()
    except Exception:
        raise ValueError("Invalid host in URL")


def is_authorized_redirect_host(hostname: str, authorized_domains: list[str]) -> bool:
    """
    Pure function that determines if a hostname is authorized for redirection.
    - Normalizes both the input hostname and the authorized domains to IDNA ASCII.
    - Allows exact matches and subdomain matches (e.g., sub.domain.com for domain.com).
    """
    host_idna = normalize_hostname_idna(hostname)
    allowed_idna = [
        d.encode("idna").decode("ascii").rstrip(".").lower()
        for d in authorized_domains
    ]
    return any(host_idna == d or host_idna.endswith("." + d) for d in allowed_idna)


def process_url_redirect(input_url: str) -> str:
    """
    Validates and sanitizes a user-provided URL for redirection.
    - Ensures absolute URL with http/https scheme.
    - Ensures hostname belongs to an authorized domain (exact match or subdomain).
    - Rejects URLs containing userinfo (username/password).
    - Returns a normalized, safe URL string.

    Logs:
        - Each redirect attempt with the input URL.
        - Successful validation with normalized details.
        - Errors encountered during processing.

    Raises:
        ValueError: If URL is invalid or its domain is not authorized.
    """
    logger.info("Redirect attempt: input_url=%s", input_url)
    try:
        if not isinstance(input_url, str):
            raise ValueError("URL must be a string")

        parsed = urlparse(input_url)

        # Must have scheme and netloc
        if parsed.scheme.lower() not in ("http", "https"):
            raise ValueError("Only http and https URLs are allowed")
        if not parsed.netloc:
            raise ValueError("URL must be absolute and include a network location")

        # Disallow userinfo to avoid phishing/ambiguous URLs
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("Userinfo in URL is not allowed")

        hostname = parsed.hostname  # lowercased by urlparse
        if not hostname:
            raise ValueError("URL must include a valid host")

        # Normalize and validate host against allowlist
        host_idna = normalize_hostname_idna(hostname)
        if not is_authorized_redirect_host(hostname, AUTHORIZED_REDIRECT_DOMAINS):
            raise ValueError("Domain is not authorized for redirection")

        # Reconstruct a sanitized URL:
        scheme = parsed.scheme.lower()

        # Preserve explicit port if present and valid
        netloc = host_idna
        if parsed.port is not None:
            if not (0 < parsed.port <= 65535):
                raise ValueError("Invalid port number")
            netloc = f"{host_idna}:{parsed.port}"

        # Sanitize path, params, query, fragment
        safe_path = quote(parsed.path, safe="/:@-._~!$&'()*+,;=")
        safe_params = quote(parsed.params, safe=":@-._~!$&'()*+,;=")

        # Normalize and encode query preserving duplicates
        query_pairs = parse_qsl(parsed.query, keep_blank_values=True, strict_parsing=False)
        safe_query = urlencode(query_pairs, doseq=True, safe=":@-._~!$&'()*+,;=")

        safe_fragment = quote(parsed.fragment, safe=":@-._~!$&'()*+,;=")

        sanitized = urlunparse((scheme, netloc, safe_path, safe_params, safe_query, safe_fragment))

        # Final guard against CRLF or other control chars
        if any(ch in sanitized for ch in ("\r", "\n")):
            raise ValueError("URL contains invalid control characters")

        logger.info(
            "Redirect validated: input_url=%s host=%s scheme=%s port=%s sanitized_url=%s",
            input_url,
            host_idna,
            scheme,
            parsed.port,
            sanitized,
        )
        return sanitized

    except ValueError as e:
        logger.warning("Redirect failed (validation): input_url=%s error=%s", input_url, e)
        raise
    except Exception:
        logger.exception("Redirect failed (unexpected error): input_url=%s", input_url)
        raise
