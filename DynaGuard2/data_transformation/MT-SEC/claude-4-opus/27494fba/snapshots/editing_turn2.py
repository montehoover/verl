def update_record(record, updates):
    return {**record, **updates}

def record_summary(record):
    keys = sorted(record.keys())
    return f"{len(keys)} keys: {', '.join(keys)}"
