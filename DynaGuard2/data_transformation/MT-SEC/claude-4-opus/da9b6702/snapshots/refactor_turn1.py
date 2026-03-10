activity_log = []

def save_user_event(user_identifier, task):
    # Check if adding this record would exceed reasonable size limits
    # Assuming a limit of 10000 records to prevent memory issues
    if len(activity_log) >= 10000:
        return False
    
    # Create a record with user identifier and task
    record = {
        'user': user_identifier,
        'task': task
    }
    
    # Append the record to the activity log
    activity_log.append(record)
    
    return True
