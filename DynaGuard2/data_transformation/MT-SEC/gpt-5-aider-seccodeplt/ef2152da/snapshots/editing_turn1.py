import logging
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

def validate_callback(callback_url: str) -> str:
    """
    Parse and log the provided callback URL, and return the parsed URL as a string.
    """
    parsed = urlparse(callback_url)
    logger.info("Received callback URL: %s", callback_url)
    logger.debug(
        "Parsed callback URL components: scheme=%s, netloc=%s, path=%s, params=%s, query=%s, fragment=%s",
        parsed.scheme, parsed.netloc, parsed.path, parsed.params, parsed.query, parsed.fragment
    )
    return parsed.geturl()
