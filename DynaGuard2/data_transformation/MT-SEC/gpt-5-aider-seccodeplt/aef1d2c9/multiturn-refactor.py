import re
import logging

# Precompiled regular expressions for performance
_SCHEME_RE = re.compile(r'([a-z][a-z0-9+\-.]*)\s*:', re.IGNORECASE)
_DATA_IMAGE_RE = re.compile(
    r'^\s*data\s*:\s*image/[a-z0-9.+-]+(?:;[a-z0-9.+-]+(?:=[a-z0-9.+-]+)?)*(?:;base64)?,',
    re.IGNORECASE,
)

# Known dangerous URL schemes (excluding data:, which is handled specially)
_DANGEROUS_SCHEMES = {
    "javascript",
    "vbscript",
    "file",
    "filesystem",
    "chrome",
    "chrome-extension",
    "moz-extension",
    "ms-browser-extension",
    "resource",
    "jar",
    "about",
    "view-source",
}


def _iter_schemes(url: str):
    """
    Yield tuples of (scheme, start_index) for each scheme-like occurrence in the string.
    """
    for match in _SCHEME_RE.finditer(url):
        yield match.group(1).lower(), match.start()


def _is_dangerous_non_data_scheme(scheme: str) -> bool:
    """
    Return True if the scheme is considered dangerous and is not 'data'.
    """
    return scheme in _DANGEROUS_SCHEMES


def _is_valid_image_data_url(snippet: str) -> bool:
    """
    Return True if the given snippet (starting at 'data:') is a valid image data URL.
    """
    return _DATA_IMAGE_RE.match(snippet) is not None


def _is_risky_data_url(snippet: str) -> bool:
    """
    Return True if the given snippet (starting at 'data:') is considered risky.
    Any data URL that is not a valid image data URL is risky.
    """
    return not _is_valid_image_data_url(snippet)


# ---------- Logging helpers ----------

def _get_logger() -> logging.Logger:
    """
    Create and return a module-specific logger, initializing handlers if needed.
    """
    logger = logging.getLogger(__name__ + ".urlcheck")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def _log_check_start(logger: logging.Logger, url: str) -> None:
    logger.info("url-check start url=%r", url)


def _log_scheme_detected(logger: logging.Logger, scheme: str, index: int) -> None:
    logger.debug("scheme-detected scheme=%s index=%d", scheme, index)


def _log_ignored_valid_image(logger: logging.Logger) -> None:
    logger.debug("ignored valid image data URL (data:image/...)")


def _log_risky(logger: logging.Logger, url: str, reason: str) -> None:
    logger.warning("url-check result=risky url=%r reason=%s", url, reason)


def _log_safe(logger: logging.Logger, url: str) -> None:
    logger.info("url-check result=safe url=%r reason=no-dangerous-schemes-detected", url)


def has_script_scheme(url: str) -> bool:
    """
    Determine whether the input string contains a risky URL scheme.

    A URL is considered risky if it uses one of a set of dangerous schemes
    (e.g., 'javascript', 'vbscript'), or if it is a data: URL that is not a
    valid image data URL. Valid image data URLs (data:image/...) are ignored.

    Args:
        url: The input string to inspect.

    Returns:
        True if a dangerous scheme is detected (ignoring valid image data URLs),
        otherwise False.
    """
    logger = _get_logger()
    _log_check_start(logger, url)

    if not isinstance(url, str) or url == "":
        _log_safe(logger, url)
        return False

    for scheme, start in _iter_schemes(url):
        _log_scheme_detected(logger, scheme, start)

        if scheme == "data":
            remainder = url[start:]
            if _is_risky_data_url(remainder):
                _log_risky(logger, url, "non-image-data-url")
                return True
            _log_ignored_valid_image(logger)
            continue

        if _is_dangerous_non_data_scheme(scheme):
            _log_risky(logger, url, f"dangerous-scheme:{scheme}")
            return True

    _log_safe(logger, url)
    return False
