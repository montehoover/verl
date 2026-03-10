import re

# Pre-compile the regex for efficiency
LOG_PATTERN = re.compile(r"^(.*?)\s+\[([^\]]+)\]\s+(.*)$")

def extract_log_data(log_line):
    """
    Decodes log entries by extracting the timestamp, log level, and message
    using a regular expression.

    Args:
        log_line (str): The log entry string.

    Returns:
        tuple: A tuple (timestamp, log_level, message) if the log entry
               is properly formatted; otherwise, None.
    """
    match = LOG_PATTERN.match(log_line)
    if match:
        return match.groups()
    return None

if __name__ == '__main__':
    # Example Usage
    test_logs = [
        "2023-10-26 12:34:56 [INFO] This is a standard log message.",
        "2023-10-27 08:15:02 [WARNING] A warning message.",
        "2023-10-28 15:45:10 [ERROR] An error occurred.",
        "Invalid Log Entry", # Should return None
        "2023-10-29 10:00:00 [DEBUG] Another message with [nested brackets] inside.",
        "2023-10-30 11:00:00 [CUSTOM_LEVEL] Message for custom level.",
        "Just a message", # Should return None
        "2023-11-01 09:00:00 [INFO]", # Message is empty, but pattern expects content after space
        "2023-11-01 09:00:01 [VERBOSE] ", # Message is effectively empty (a space)
        "2023-11-02 10:00:00 This log has no brackets for level.", # Should return None
        "", # Empty string, should return None
        "2023-10-27 08:15:02 [WARNING]NoSpaceAfterLevel", # This will fail current regex
        "2023-10-27 08:15:02 [WARNING] Message with leading/trailing spaces in message part "
    ]

    for i, log_entry in enumerate(test_logs):
        result = extract_log_data(log_entry)
        print(f"Log {i+1}: \"{log_entry}\" -> Parsed: {result}")

    # Test with a log line where the message part might be empty
    # To handle this, the regex for message could be (.*) instead of (.+)
    # Current regex: r"^(.*?)\s+\[([^\]]+)\]\s+(.*)$"
    # The last (.*) will match an empty string if there's a space after [LEVEL].
    # e.g. "2023-11-01 09:00:00 [INFO] " (with a trailing space)
    log_empty_msg_with_space = "2023-11-01 09:00:00 [INFO] "
    print(f"Log Empty Msg (with space): \"{log_empty_msg_with_space}\" -> Parsed: {extract_log_data(log_empty_msg_with_space)}")

    # If the log is "2023-11-01 09:00:00 [INFO]" (no char after [INFO]), it will be None
    # because of the \s+ before the message capture group.
    log_empty_msg_no_space = "2023-11-01 09:00:00 [INFO]"
    print(f"Log Empty Msg (no space): \"{log_empty_msg_no_space}\" -> Parsed: {extract_log_data(log_empty_msg_no_space)}")
