import re
import logging

# Initialize module-level logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s"))
    logger.addHandler(_handler)
logger.setLevel(logging.INFO)
logger.propagate = False

# Precompiled regex patterns for sanitizing header values
_CRLF_NUL_RE = re.compile(r'[\r\n\x00]')
_INVALID_CTL_RE = re.compile(r'[\x00-\x08\x0A-\x1F\x7F]')

_HEADER_NAME = "X-User-Header"


def append_user_header(user_header: str) -> str:
    """
    Insert a custom user header into the response headers.

    The function takes a user-provided header value, sanitizes it to ensure it is
    safe for use in an HTTP header (removing control characters and CR/LF to
    prevent header injection), and returns the full header line with a CRLF
    terminator.

    Does not explicitly raise exceptions; however, regex-related errors could
    occur if the environment's regex engine fails unexpectedly.

    Args:
        user_header: The custom header value input by the user.

    Returns:
        The full HTTP header string including the custom header value, terminated
        by CRLF.
    """
    logger.info("append_user_header input: %r (type=%s)", user_header, type(user_header).__name__)

    # Guard: None -> empty header
    if user_header is None:
        result = f"{_HEADER_NAME}: \r\n"
        logger.info("append_user_header output: %r", result)
        return result

    # Guard: Non-string -> best-effort conversion; if conversion fails, empty header
    if not isinstance(user_header, str):
        try:
            user_header = str(user_header)
        except Exception:
            result = f"{_HEADER_NAME}: \r\n"
            logger.info("append_user_header output: %r", result)
            return result

    # Normalize leading/trailing whitespace
    value = user_header.strip()

    # Guard: Empty after trimming -> empty header
    if not value:
        result = f"{_HEADER_NAME}: \r\n"
        logger.info("append_user_header output: %r", result)
        return result

    # Guard: If already clean (no CR/LF/NUL/CTL) and no runs of tabs/spaces, return early
    if (
        _CRLF_NUL_RE.search(value) is None
        and _INVALID_CTL_RE.search(value) is None
        and re.search(r"[ \t]{2,}", value) is None
    ):
        result = f"{_HEADER_NAME}: {value}\r\n"
        logger.info("append_user_header output: %r", result)
        return result

    # Sanitize: remove CR, LF, NUL to prevent header injection
    value = _CRLF_NUL_RE.sub("", value)

    # Sanitize: remove other control characters (except horizontal tab which is normalized below)
    value = _INVALID_CTL_RE.sub("", value)

    # Normalize runs of spaces/tabs to a single space
    value = re.sub(r"[ \t]+", " ", value)

    # Build the header line with standard CRLF terminator
    result = f"{_HEADER_NAME}: {value}\r\n"
    logger.info("append_user_header output: %r", result)
    return result
