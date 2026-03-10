import datetime

LOG_FILE = "system_events.log"

def log_system_event(event_type: str, event_description: str) -> bool:
    """
    Logs a system event to a file.

    Args:
        event_type: The type of the event (e.g., "INFO", "ERROR", "WARNING").
        event_description: A description of the event.

    Returns:
        True if the event was logged successfully, False otherwise.
    """
    try:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"{timestamp} [{event_type}] {event_description}\n"
        with open(LOG_FILE, "a") as f:
            f.write(log_entry)
        return True
    except IOError:
        # Optionally, print an error message to stderr or handle the error in another way
        # print(f"Error: Could not write to log file {LOG_FILE}")
        return False

if __name__ == '__main__':
    # Example usage:
    if log_system_event("INFO", "Application started successfully."):
        print("Event logged.")
    else:
        print("Failed to log event.")

    if log_system_event("ERROR", "Failed to connect to database."):
        print("Event logged.")
    else:
        print("Failed to log event.")
