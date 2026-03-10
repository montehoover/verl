event_log = []

def log_event(user_id: str, event_description: str) -> bool:
    if len(event_log) >= 100:
        return False
    try:
        event_log.append({"user_id": user_id, "event_description": event_description})
        return True
    except Exception:
        return False

ACTIVITY_LOG_MAX_SIZE = 100
activity_log = []

def save_user_event(user_identifier: str, task: str) -> bool:
    if len(activity_log) >= ACTIVITY_LOG_MAX_SIZE:
        return False
    try:
        activity_log.append({"user_identifier": user_identifier, "task": task})
        return True
    except Exception:
        return False
