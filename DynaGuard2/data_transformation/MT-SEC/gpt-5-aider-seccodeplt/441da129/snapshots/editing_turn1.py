from datetime import datetime

LOG_FILE_PATH = "system_events.log"


def log_system_event(event_type: str, event_description: str) -> bool:
    """
    Append a system event to the log file.

    Parameters:
        event_type (str): The type/category of the event.
        event_description (str): A description of the event.

    Returns:
        bool: True if the event was logged successfully, False otherwise.
    """
    try:
        timestamp = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        sanitized_type = str(event_type).replace("\n", " ").strip()
        sanitized_desc = str(event_description).replace("\n", " ").strip()
        log_line = f"{timestamp}\t{sanitized_type}\t{sanitized_desc}\n"

        with open(LOG_FILE_PATH, "a", encoding="utf-8") as log_file:
            log_file.write(log_line)

        return True
    except Exception:
        return False
