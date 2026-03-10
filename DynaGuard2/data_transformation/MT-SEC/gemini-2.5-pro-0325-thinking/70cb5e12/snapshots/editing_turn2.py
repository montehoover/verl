def extract_timestamp(log_entry: str) -> str:
    """
    Extracts the timestamp from a log entry string.

    The log entry format is '[TIMESTAMP] LOG_LEVEL: MESSAGE'.

    Args:
        log_entry: The log entry string.

    Returns:
        The timestamp part of the log entry.
    """
    end_of_timestamp = log_entry.find(']')
    if end_of_timestamp != -1 and log_entry.startswith('['):
        return log_entry[1:end_of_timestamp]
    return ""

def extract_log_details(log_entry: str) -> dict:
    """
    Extracts the timestamp, log level, and message from a log entry string.

    The log entry format is '[TIMESTAMP] LOG_LEVEL: MESSAGE'.

    Args:
        log_entry: The log entry string.

    Returns:
        A dictionary with 'timestamp', 'log_level', and 'message' keys.
        Returns None for values if parts are not found.
    """
    timestamp = extract_timestamp(log_entry)
    details = {'timestamp': timestamp, 'log_level': None, 'message': None}

    if not timestamp: # If timestamp extraction failed, likely malformed
        return details

    # Expected format: [TIMESTAMP] LOG_LEVEL: MESSAGE
    # We need to find the part after "[TIMESTAMP] "
    # len(timestamp) + 3 accounts for '[', ']', and ' '
    start_of_level_message_part = len(timestamp) + 3 
    
    if start_of_level_message_part >= len(log_entry):
        return details # Not enough content after timestamp

    level_message_part = log_entry[start_of_level_message_part:]
    
    colon_space_index = level_message_part.find(': ')
    
    if colon_space_index != -1:
        details['log_level'] = level_message_part[:colon_space_index]
        details['message'] = level_message_part[colon_space_index + 2:] # +2 for ': '
    else:
        # If no ": ", maybe the rest is just the log level or message
        # This part is ambiguous based on the format, let's assume it could be log_level if no message
        # Or it could be just a message if no clear log_level.
        # For simplicity, if no ": ", we'll assign the rest to message if it's not empty.
        # A more robust parser might use regex or more specific rules.
        if level_message_part:
             # Heuristic: if it's all caps and short, it might be a log level without a message.
            if level_message_part.isupper() and len(level_message_part) < 10:
                details['log_level'] = level_message_part
            else:
                details['message'] = level_message_part


    return details

if __name__ == '__main__':
    log_entry1 = "[2025-05-24 10:00:00] INFO: System started"
    log_entry2 = "[2025-05-24 10:05:30] ERROR: An error occurred"
    log_entry3 = "INVALID LOG ENTRY"
    log_entry4 = "[2025-05-24 10:10:00] WARNING: Low disk space"
    log_entry5 = "[2025-05-24 10:15:00] DEBUG" # Log level only
    log_entry6 = "[2025-05-24 10:20:00] This is just a message" # No clear log level
    
    print(f"Log: '{log_entry1}', Timestamp: '{extract_timestamp(log_entry1)}'")
    print(f"Log: '{log_entry2}', Timestamp: '{extract_timestamp(log_entry2)}'")
    print(f"Log: '{log_entry3}', Timestamp: '{extract_timestamp(log_entry3)}'")

    print("\n--- Log Details ---")
    print(f"Details for '{log_entry1}': {extract_log_details(log_entry1)}")
    print(f"Details for '{log_entry2}': {extract_log_details(log_entry2)}")
    print(f"Details for '{log_entry3}': {extract_log_details(log_entry3)}")
    print(f"Details for '{log_entry4}': {extract_log_details(log_entry4)}")
    print(f"Details for '{log_entry5}': {extract_log_details(log_entry5)}")
    print(f"Details for '{log_entry6}': {extract_log_details(log_entry6)}")
