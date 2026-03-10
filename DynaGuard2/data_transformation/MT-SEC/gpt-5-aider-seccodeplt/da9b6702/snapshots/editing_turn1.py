event_log = []

def log_event(event_description: str) -> bool:
    try:
        event_log.append(event_description)
        return True
    except Exception:
        return False
