import re


# Module-level constant for the compiled regular expression pattern.
# This pattern is designed to match log entries in the format:
# '[TIMESTAMP] LOG_LEVEL: MESSAGE'
LOG_ENTRY_PATTERN = re.compile(r"\[(.*?)\]\s(.*?):\s(.*)")


def _parse_log_entry(record: str, pattern: re.Pattern):
    """
    Matches the log record against the given pre-compiled regex pattern.

    This is a pure helper function designed to encapsulate the regex matching logic.

    Args:
        record: The log entry string to be parsed.
        pattern: The compiled regex pattern to use for matching.

    Returns:
        A regex match object if the record string matches the pattern,
        otherwise None.
    """
    return pattern.match(record)


def _extract_log_components(match_obj: re.Match):
    """
    Extracts timestamp, log_level, and message from a regex match object.

    This is a pure helper function that assumes a valid match object is provided,
    typically obtained from `_parse_log_entry`.

    Args:
        match_obj: A regex match object resulting from a successful pattern match.
                   It is expected to contain three capturing groups.

    Returns:
        A tuple containing the extracted (timestamp, log_level, message).
    """
    # The groups() method returns a tuple of all subgroups of the match.
    # For our pattern, these are:
    # 1. Timestamp (e.g., "2023-10-26T10:00:00")
    # 2. Log Level (e.g., "INFO", "ERROR")
    # 3. Message (e.g., "User logged in", "Failed to connect to database")
    return match_obj.groups()


def analyze_log_data(record: str):
    """
    Decodes a single log entry string to extract its structured components.

    This function orchestrates the parsing and extraction process. It uses
    helper functions to match the log entry against a predefined pattern
    and then to extract the relevant parts if the pattern matches.

    The expected log entry format is '[TIMESTAMP] LOG_LEVEL: MESSAGE'.

    Args:
        record: A string representing a single log entry.

    Returns:
        A tuple (timestamp, log_level, message) if the log entry
        is properly formatted and matches the expected pattern.
        Otherwise, if the log entry does not match or is malformed,
        it returns None. No exceptions are raised for parsing failures.
    """
    match = _parse_log_entry(record, LOG_ENTRY_PATTERN)

    if match:
        return _extract_log_components(match)
    
    return None
