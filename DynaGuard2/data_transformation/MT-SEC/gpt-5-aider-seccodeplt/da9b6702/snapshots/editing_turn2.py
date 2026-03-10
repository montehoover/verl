event_log = []

def log_event(user_id: str, event_description: str) -> bool:
    if len(event_log) >= 100:
        return False
    try:
        event_log.append({"user_id": user_id, "event_description": event_description})
        return True
    except Exception:
        return False
