import logging
import re
from typing import Optional, Any

logger = logging.getLogger(__name__)

EMAIL_DOMAIN_REGEX = re.compile(
    r'^[^@\s]+@(?P<domain>[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+)$'
)


def _normalize_email(email_address: Any) -> Optional[str]:
    """
    Return a stripped string if the input is a non-empty string; otherwise, None.
    Pure function: no side effects.
    """
    if not isinstance(email_address, str):
        return None
    normalized = email_address.strip()
    return normalized or None


def _extract_domain(email: str) -> Optional[str]:
    """
    Return the domain if the email matches the expected pattern; otherwise, None.
    Pure function: no side effects.
    """
    match = EMAIL_DOMAIN_REGEX.match(email)
    return match.group('domain') if match else None


def retrieve_domain_from_email(email_address: str) -> Optional[str]:
    """
    Retrieve the domain from a valid email address using a regular expression.

    Args:
        email_address: The email address to process and analyze.

    Returns:
        The domain portion if the email is valid; otherwise, None.
    """
    logger.debug("Starting domain retrieval for input: %r", email_address)

    normalized = _normalize_email(email_address)
    if normalized is None:
        if not isinstance(email_address, str):
            logger.warning(
                "Invalid email input: expected str but received %s",
                type(email_address).__name__
            )
        else:
            logger.debug("Invalid email input: empty after stripping whitespace.")
        return None

    domain = _extract_domain(normalized)
    if domain is None:
        logger.debug("Email did not match expected pattern: %s", normalized)
        return None

    logger.info("Successfully extracted domain '%s' from email '%s'", domain, normalized)
    return domain
