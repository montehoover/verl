def identify_log_parts(log_entry):
    """
    Identifies and extracts timestamp, log level, and message from a log entry string.

    Assumes a log format where the log level is enclosed in square brackets, e.g.,
    "YYYY-MM-DD HH:MM:SS [LEVEL] Message content"
    Other delimiters like colons are not explicitly used for splitting in this version,
    but the logic primarily relies on the square brackets for the log level.

    Args:
        log_entry (str): The log entry string.

    Returns:
        dict: A dictionary with keys 'timestamp', 'log_level', and 'message'.
              Values might be None or empty strings if parts are not found or
              parsing fails.
    """
    log_parts = {"timestamp": None, "log_level": None, "message": None}

    try:
        level_start_marker_idx = log_entry.find('[')
        level_end_marker_idx = log_entry.find(']', level_start_marker_idx)

        if level_start_marker_idx != -1 and level_end_marker_idx != -1:
            log_parts["timestamp"] = log_entry[:level_start_marker_idx].strip()
            log_parts["log_level"] = log_entry[level_start_marker_idx + 1:level_end_marker_idx]
            log_parts["message"] = log_entry[level_end_marker_idx + 1:].strip()
        elif level_start_marker_idx != -1:  # Found '[' but no ']'
            # Assume timestamp is before '[' and rest is message (or malformed level)
            log_parts["timestamp"] = log_entry[:level_start_marker_idx].strip()
            log_parts["message"] = log_entry[level_start_marker_idx:].strip()  # Treat rest as message
        else:
            # No '[]' found, try a simple split by space for timestamp, level, message
            # This is a fallback and might not be accurate for all formats
            components = log_entry.split(None, 2)
            if len(components) == 3:
                log_parts["timestamp"] = components[0]
                log_parts["log_level"] = components[1]
                log_parts["message"] = components[2]
            elif len(components) == 2:
                log_parts["timestamp"] = components[0]
                log_parts["message"] = components[1]  # Assume no level, or level is part of message
            elif len(components) == 1:
                log_parts["message"] = components[0]  # Assume whole string is message
            # If components is empty, log_parts remains with None values for unparsed fields

    except Exception:
        # In case of any unexpected error during string operations,
        # return whatever parts could be parsed, or default None/empty parts.
        # This makes the function more resilient.
        pass

    # Ensure all keys exist, even if values are empty strings for consistency if preferred over None
    for key in ["timestamp", "log_level", "message"]:
        if log_parts[key] is None:
            log_parts[key] = ""


    return log_parts

if __name__ == '__main__':
    # Example Usage
    log1 = "2023-10-26 12:34:56 [INFO] This is a standard log message."
    log2 = "2023-10-27 08:15:02 [WARNING]Something went slightly wrong."
    log3 = "2023-10-28 15:45:10 ERROR This log has no brackets for level."
    log4 = "Invalid Log Entry"
    log5 = "2023-10-29 10:00:00 [DEBUG] Another message with [nested brackets] inside."
    log6 = "2023-10-30 11:00:00 [CUSTOM_LEVEL_WITH_SPACES] Message for custom level."
    log7 = "Just a message"
    log8 = "2023-11-01 09:00:00 [INFO]" # Message is empty
    log9 = "2023-11-02 10:00:00 This might be a timestamp and a message"
    log10 = "" # Empty string

    print(f"Log: \"{log1}\" -> Parsed: {identify_log_parts(log1)}")
    print(f"Log: \"{log2}\" -> Parsed: {identify_log_parts(log2)}")
    print(f"Log: \"{log3}\" -> Parsed: {identify_log_parts(log3)}")
    print(f"Log: \"{log4}\" -> Parsed: {identify_log_parts(log4)}")
    print(f"Log: \"{log5}\" -> Parsed: {identify_log_parts(log5)}")
    print(f"Log: \"{log6}\" -> Parsed: {identify_log_parts(log6)}")
    print(f"Log: \"{log7}\" -> Parsed: {identify_log_parts(log7)}")
    print(f"Log: \"{log8}\" -> Parsed: {identify_log_parts(log8)}")
    print(f"Log: \"{log9}\" -> Parsed: {identify_log_parts(log9)}")
    print(f"Log: \"{log10}\" -> Parsed: {identify_log_parts(log10)}")

    # Example with a different structure (no brackets, space separated)
    log_space_separated = "2024-01-01T12:00:00Z DEBUG This is a debug message."
    print(f"Log: \"{log_space_separated}\" -> Parsed: {identify_log_parts(log_space_separated)}")

    # Example with missing closing bracket
    log_missing_bracket = "2024-01-02T14:30:00Z [ERROR This message is malformed."
    print(f"Log: \"{log_missing_bracket}\" -> Parsed: {identify_log_parts(log_missing_bracket)}")
