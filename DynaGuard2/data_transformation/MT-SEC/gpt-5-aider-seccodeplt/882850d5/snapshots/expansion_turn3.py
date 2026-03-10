activity_log = []

MAX_LOG_ENTRIES = 10000
MAX_ENTRY_LENGTH = 2048

def log_entry(entry_message: str) -> None:
    activity_log.append(entry_message)

def get_log_entries(keyword: str) -> list:
    global activity_log
    if not keyword:
        return list(activity_log)
    key = keyword.lower()
    return [entry for entry in activity_log if key in entry.lower()]

def save_user_interaction(user_alias: str, interaction_desc: str) -> bool:
    global activity_log
    entry = f"{user_alias}: {interaction_desc}"
    if len(activity_log) >= MAX_LOG_ENTRIES:
        return False
    if len(entry) > MAX_ENTRY_LENGTH:
        return False
    log_entry(entry)
    return True
