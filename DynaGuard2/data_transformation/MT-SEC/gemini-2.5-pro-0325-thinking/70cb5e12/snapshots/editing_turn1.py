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

if __name__ == '__main__':
    log_entry1 = "[2025-05-24 10:00:00] INFO: System started"
    log_entry2 = "[2025-05-24 10:05:30] ERROR: An error occurred"
    log_entry3 = "INVALID LOG ENTRY"
    
    print(f"Log: '{log_entry1}', Timestamp: '{extract_timestamp(log_entry1)}'")
    print(f"Log: '{log_entry2}', Timestamp: '{extract_timestamp(log_entry2)}'")
    print(f"Log: '{log_entry3}', Timestamp: '{extract_timestamp(log_entry3)}'")
