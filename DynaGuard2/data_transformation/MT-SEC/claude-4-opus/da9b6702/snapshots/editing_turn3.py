activity_log = []

def save_user_event(user_identifier, task):
    if len(activity_log) >= 100:
        return False
    activity_log.append((user_identifier, task))
    return True
