from datetime import datetime, timezone

LOG_FILE_PATH = "system_events.log"


def log_system_event(event_type: str, event_description: str, event_timestamp) -> bool:
    """
    Append a system event to the log file.

    Parameters:
        event_type (str): The type/category of the event.
        event_description (str): A description of the event.
        event_timestamp: When the event occurred. Accepts:
            - datetime (naive assumed UTC, aware converted to UTC)
            - str (ISO 8601; 'Z' suffix supported)
            - int/float (Unix timestamp seconds)

    Returns:
        bool: True if the event was logged successfully, False otherwise.
    """
    try:
        # Sanitize text fields to keep log line structure intact
        sanitized_type = str(event_type).replace("\n", " ").replace("\t", " ").strip()
        sanitized_desc = str(event_description).replace("\n", " ").replace("\t", " ").strip()

        # Normalize/format timestamp to ISO 8601 UTC with 'Z' suffix, seconds precision
        ts = event_timestamp
        timestamp_str = None

        if isinstance(ts, datetime):
            dt = ts
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            else:
                dt = dt.astimezone(timezone.utc)
            timestamp_str = dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")
        elif isinstance(ts, (int, float)):
            dt = datetime.fromtimestamp(float(ts), tz=timezone.utc)
            timestamp_str = dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")
        elif isinstance(ts, str):
            s = ts.strip()
            if not s:
                return False
            try:
                s2 = s[:-1] + "+00:00" if s.endswith("Z") else s
                dt = datetime.fromisoformat(s2)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                else:
                    dt = dt.astimezone(timezone.utc)
                timestamp_str = dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")
            except Exception:
                return False
        else:
            return False

        # Final guard against control characters
        timestamp_str = timestamp_str.replace("\n", " ").replace("\t", " ").strip()

        log_line = f"{timestamp_str}\t{sanitized_type}\t{sanitized_desc}\n"

        with open(LOG_FILE_PATH, "a", encoding="utf-8") as log_file:
            log_file.write(log_line)

        return True
    except Exception:
        return False
