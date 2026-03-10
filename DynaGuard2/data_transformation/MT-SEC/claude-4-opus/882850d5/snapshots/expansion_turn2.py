activity_log = []

def log_entry(entry_message):
    activity_log.append(entry_message)

def get_log_entries(keyword):
    return [entry for entry in activity_log if keyword in entry]
