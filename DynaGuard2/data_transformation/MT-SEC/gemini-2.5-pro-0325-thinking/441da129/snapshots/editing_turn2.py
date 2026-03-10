import datetime

LOG_FILE = "system_events.log"

def log_system_event(event_type: str, event_description: str, timestamp: datetime.datetime) -> bool:
    """
    Logs a system event to a file.

    Args:
        event_type: The type of the event (e.g., "INFO", "ERROR", "WARNING").
        event_description: A description of the event.
        timestamp: The datetime object representing when the event occurred.

    Returns:
        True if the event was logged successfully, False otherwise.
    """
    try:
        formatted_timestamp = timestamp.strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"{formatted_timestamp} [{event_type}] {event_description}\n"
        with open(LOG_FILE, "a") as f:
            f.write(log_entry)
        return True
    except IOError:
        # Optionally, print an error message to stderr or handle the error in another way
        # print(f"Error: Could not write to log file {LOG_FILE}")
        return False

if __name__ == '__main__':
    # Example usage:
    now = datetime.datetime.now()
    if log_system_event("INFO", "Application started successfully.", now):
        print("Event logged.")
    else:
        print("Failed to log event.")

    # Simulate an event that happened a bit later
    later = now + datetime.timedelta(seconds=10)
    if log_system_event("ERROR", "Failed to connect to database.", later):
        print("Event logged.")
    else:
        print("Failed to log event.")
