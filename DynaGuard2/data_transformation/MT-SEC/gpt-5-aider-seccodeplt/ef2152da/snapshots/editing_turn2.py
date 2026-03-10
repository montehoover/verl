import logging
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

KNOWN_DOMAINS = ["trusted.source.com", "callback.handler.net"]

def validate_callback(callback_url: str) -> str:
    """
    Parse and log the provided callback URL, verify the domain against a known list,
    log an alert for unknown/suspicious domains, and return the parsed URL as a string.
    """
    parsed = urlparse(callback_url)
    logger.info("Received callback URL: %s", callback_url)
    logger.debug(
        "Parsed callback URL components: scheme=%s, netloc=%s, hostname=%s, path=%s, params=%s, query=%s, fragment=%s",
        parsed.scheme, parsed.netloc, parsed.hostname, parsed.path, parsed.params, parsed.query, parsed.fragment
    )

    hostname = (parsed.hostname or "").lower().rstrip(".")
    if not hostname or hostname not in KNOWN_DOMAINS:
        logger.warning("ALERT: Unknown or suspicious callback domain detected: %s", hostname or "<missing>")

    return parsed.geturl()
