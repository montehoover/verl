activity_log = []

def log_entry(entry_message: str) -> None:
    activity_log.append(entry_message)

def get_log_entries(keyword: str) -> list:
    global activity_log
    if not keyword:
        return list(activity_log)
    key = keyword.lower()
    return [entry for entry in activity_log if key in entry.lower()]
