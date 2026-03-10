activity_log = []

def is_log_full(log, max_size=10000):
    """Check if the log has reached its maximum size."""
    return len(log) >= max_size

def create_activity_record(user_identifier, task):
    """Create a new activity record."""
    return {
        'user': user_identifier,
        'task': task
    }

def append_to_log(log, record):
    """Append a record to the log."""
    log.append(record)

def save_user_event(user_identifier, task):
    # Check if adding this record would exceed reasonable size limits
    if is_log_full(activity_log):
        return False
    
    # Create a record with user identifier and task
    record = create_activity_record(user_identifier, task)
    
    # Append the record to the activity log
    append_to_log(activity_log, record)
    
    return True
