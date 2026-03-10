import re

# Regex to capture timestamp, log_level, and message
# Timestamp: anything within square brackets
# Log Level: one or more word characters (letters, numbers, underscore)
# Message: anything after 'LOG_LEVEL: '
LOG_ENTRY_PATTERN = re.compile(r"\[(.*?)\]\s+(\w+):\s+(.*)")


def _match_log_entry(log_line: str) -> re.Match | None:
    """Match a log line against the predefined log entry pattern.

    This utility function attempts to match the given `log_line` using
    the compiled `LOG_ENTRY_PATTERN`. It is designed to be robust and
    avoid raising exceptions.

    Args:
        log_line: The log entry string to match.

    Returns:
        A `re.Match` object if the pattern successfully matches the
        beginning of the `log_line`. If no match is found or if an
        exception occurs during the matching process, `None` is returned.
    """
    try:
        return LOG_ENTRY_PATTERN.match(log_line)
    except Exception:
        # Catch any unexpected error during regex matching to ensure robustness.
        return None


def extract_log_data(log_line: str):
    """Decode log entries by extracting timestamp, level, and message.

    This function parses a log entry string, expecting it to follow the
    format: '[TIMESTAMP] LOG_LEVEL: MESSAGE'. It extracts these three
    components.

    Args:
        log_line: The log entry string to be parsed.

    Returns:
        A tuple `(timestamp, log_level, message)` if parsing is successful.
        Returns `None` if the log line does not match the expected format,
        if the input is invalid, or if any other error occurs during
        processing, ensuring no exceptions are raised.
    """
    try:
        match = _match_log_entry(log_line)
        if not match:
            return None  # Guard clause for no match

        # If a match is found, extract and return the captured groups.
        timestamp, log_level, message = match.groups()
        return timestamp, log_level, message
    except Exception:
        # Catch any other unexpected exceptions (e.g., from .groups() if `match`
        # object was malformed, though `_match_log_entry` should prevent this,
        # or if `log_line` type issues arise that `_match_log_entry` didn't cover)
        # to ensure adherence to the non-raising requirement.
        return None
