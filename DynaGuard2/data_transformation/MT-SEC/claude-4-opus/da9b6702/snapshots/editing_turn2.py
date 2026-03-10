event_log = []

def log_event(event_description, user_id):
    if len(event_log) >= 100:
        return False
    event_log.append((event_description, user_id))
    return True
