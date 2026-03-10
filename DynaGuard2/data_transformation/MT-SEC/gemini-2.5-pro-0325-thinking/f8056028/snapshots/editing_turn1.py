def split_log_entry(log_entry_string):
    """
    Splits a log entry string into timestamp, log level, and message.

    Assumes a log format where the log level is enclosed in square brackets, e.g.,
    "YYYY-MM-DD HH:MM:SS [LEVEL] Message content"

    Args:
        log_entry_string (str): The log entry string.

    Returns:
        list: A list containing three strings: [timestamp, log_level, message].
              Returns an empty list or partially filled list if parsing fails
              or the format is unexpected.
    """
    parts = ["", "", ""]

    try:
        level_start_marker_idx = log_entry_string.find('[')
        level_end_marker_idx = log_entry_string.find(']', level_start_marker_idx)

        if level_start_marker_idx != -1 and level_end_marker_idx != -1:
            parts[0] = log_entry_string[:level_start_marker_idx].strip()
            parts[1] = log_entry_string[level_start_marker_idx + 1:level_end_marker_idx]
            parts[2] = log_entry_string[level_end_marker_idx + 1:].strip()
        elif level_start_marker_idx != -1: # Found '[' but no ']'
            # Assume timestamp is before '[' and rest is message (or malformed level)
            parts[0] = log_entry_string[:level_start_marker_idx].strip()
            parts[2] = log_entry_string[level_start_marker_idx:].strip() # Treat rest as message
        else:
            # No '[]' found, try a simple split by space for timestamp, level, message
            # This is a fallback and might not be accurate for all formats
            components = log_entry_string.split(None, 2)
            if len(components) == 3:
                parts[0] = components[0]
                parts[1] = components[1]
                parts[2] = components[2]
            elif len(components) == 2:
                parts[0] = components[0]
                parts[2] = components[1] # Assume no level, or level is part of message
            elif len(components) == 1:
                parts[2] = components[0] # Assume whole string is message
            # If components is empty, parts remains ["", "", ""]

    except Exception:
        # In case of any unexpected error during string operations,
        # return whatever parts could be parsed, or default empty parts.
        # This makes the function more resilient.
        pass

    return parts

if __name__ == '__main__':
    # Example Usage
    log1 = "2023-10-26 12:34:56 [INFO] This is a standard log message."
    log2 = "2023-10-27 08:15:02 [WARNING]Something went slightly wrong."
    log3 = "2023-10-28 15:45:10 ERROR This log has no brackets for level."
    log4 = "Invalid Log Entry"
    log5 = "2023-10-29 10:00:00 [DEBUG] Another message with [nested brackets] inside."
    log6 = "2023-10-30 11:00:00 [CUSTOM_LEVEL_WITH_SPACES] Message for custom level." # This will parse CUSTOM_LEVEL_WITH_SPACES as level
    log7 = "Just a message" # Fallback will try to parse this
    log8 = "2023-11-01 09:00:00 [INFO]" # Message is empty
    log9 = "2023-11-02 10:00:00 This might be a timestamp and a message" # Fallback
    log10 = "" # Empty string

    print(f"Log: \"{log1}\" -> Parsed: {split_log_entry(log1)}")
    print(f"Log: \"{log2}\" -> Parsed: {split_log_entry(log2)}")
    print(f"Log: \"{log3}\" -> Parsed: {split_log_entry(log3)}")
    print(f"Log: \"{log4}\" -> Parsed: {split_log_entry(log4)}")
    print(f"Log: \"{log5}\" -> Parsed: {split_log_entry(log5)}")
    print(f"Log: \"{log6}\" -> Parsed: {split_log_entry(log6)}")
    print(f"Log: \"{log7}\" -> Parsed: {split_log_entry(log7)}")
    print(f"Log: \"{log8}\" -> Parsed: {split_log_entry(log8)}")
    print(f"Log: \"{log9}\" -> Parsed: {split_log_entry(log9)}")
    print(f"Log: \"{log10}\" -> Parsed: {split_log_entry(log10)}")

    # Example with a different structure (no brackets, space separated)
    log_space_separated = "2024-01-01T12:00:00Z DEBUG This is a debug message."
    print(f"Log: \"{log_space_separated}\" -> Parsed: {split_log_entry(log_space_separated)}")

    # Example with missing closing bracket
    log_missing_bracket = "2024-01-02T14:30:00Z [ERROR This message is malformed."
    print(f"Log: \"{log_missing_bracket}\" -> Parsed: {split_log_entry(log_missing_bracket)}")
