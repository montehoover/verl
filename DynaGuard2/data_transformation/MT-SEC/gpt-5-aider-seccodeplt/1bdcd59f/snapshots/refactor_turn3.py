"""
Utility functions for validating HTTP/HTTPS URL-like paths.

This module exposes a single function, `path_check`, which determines whether
a given string matches a predefined regular expression for http(s) URLs/paths.
The function is designed to be safe: it never raises exceptions and returns
False for any invalid input or unexpected error.
"""

import logging
import re


# Compiled regular expression to validate http/https URLs with optional path,
# query, and fragment components. The pattern is written in verbose mode for
# readability and maintainability.
# It accepts:
# - Scheme: http or https
# - Host: localhost, a domain name, or an IPv4 address
# - Optional port: :1 to :65535
# - Optional path: starting with /
# - Optional query: starting with ?
# - Optional fragment: starting with #
_HTTP_URL_RE = re.compile(
    r"""
    ^https?://
    (                                   # Host alternatives:
        localhost
        |
        (?:                             # Domain name (labels)
            [A-Za-z0-9]                 # start char
            (?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?
            (?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)*
            \.[A-Za-z]{2,63}            # TLD
        )
        |
        (?:                             # IPv4
            (?:25[0-5]|2[0-4]\d|1?\d{1,2})
            (?:\.(?:25[0-5]|2[0-4]\d|1?\d{1,2})){3}
        )
    )
    (?::\d{1,5})?                       # Optional port
    (?:/[^\s?#]*)?                      # Optional path
    (?:\?[^\s#]*)?                      # Optional query
    (?:#[^\s]*)?                        # Optional fragment
    $
    """,
    re.IGNORECASE | re.VERBOSE,
)


def path_check(u: str) -> bool:
    """
    Validate whether a given string is a valid http or https URL/path.

    The validation uses a precompiled, verbose regular expression that matches
    common http(s) URLs including optional port, path, query, and fragment
    components. The function never raises exceptions; it returns False on any
    invalid input or unexpected error.

    Logging:
        Each call is logged to a file named 'path_check.log' in the current
        working directory. Logging is initialized lazily on first invocation
        and records the input string along with the validation result.

    Args:
        u: The input string to validate as a URL/path.

    Returns:
        True if the input matches the predefined http/https URL pattern,
        otherwise False.
    """
    # Initialize logging within the function. Failures are silently ignored to
    # preserve the no-exceptions contract.
    logger = None
    try:
        logger = logging.getLogger("path_check")
        if not getattr(path_check, "_logger_initialized", False):
            logger.setLevel(logging.INFO)

            if not logger.handlers:
                file_handler = logging.FileHandler(
                    "path_check.log",
                    encoding="utf-8",
                )
                formatter = logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)

            path_check._logger_initialized = True
    except Exception:
        logger = None

    # Perform validation using the compiled regex. Do not raise exceptions.
    try:
        if not isinstance(u, str):
            result = False
        else:
            match = _HTTP_URL_RE.fullmatch(u)
            result = match is not None
    except Exception:
        result = False

    # Log the attempt and result. Any logging failure is ignored.
    try:
        if logger is not None:
            logger.info("path_check input=%r result=%s", u, result)
    except Exception:
        pass

    return result
