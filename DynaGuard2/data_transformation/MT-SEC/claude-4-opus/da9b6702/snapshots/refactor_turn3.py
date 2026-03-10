import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

activity_log = []


def is_log_full(log, max_size=10000):
    """
    Check if the log has reached its maximum size.
    
    Args:
        log (list): The log to check.
        max_size (int): The maximum allowed size of the log. Defaults to 10000.
    
    Returns:
        bool: True if the log is full, False otherwise.
    """
    return len(log) >= max_size


def create_activity_record(user_identifier, task):
    """
    Create a new activity record.
    
    Args:
        user_identifier (str): A string that uniquely identifies the user.
        task (str): A string describing the specific action taken by the user.
    
    Returns:
        dict: A dictionary containing the user identifier and task.
    """
    return {
        'user': user_identifier,
        'task': task
    }


def append_to_log(log, record):
    """
    Append a record to the log.
    
    Args:
        log (list): The log to append to.
        record (dict): The record to append.
    
    Returns:
        None
    """
    log.append(record)


def save_user_event(user_identifier, task):
    """
    Captures and stores user activities in a system by appending new records to an activity log.
    
    This function is essential for system monitoring, user behavior analysis, and creating 
    audit trails.
    
    Args:
        user_identifier (str): A string that uniquely identifies the user performing the action.
        task (str): A string describing the specific action taken by the user.
    
    Returns:
        bool: True if the activity was successfully logged, False if the logging attempt 
              failed due to exceeding size limits.
    """
    logger.info(f"Attempting to log activity for user: {user_identifier}, task: {task}")
    
    # Check if adding this record would exceed reasonable size limits
    if is_log_full(activity_log):
        logger.warning(f"Activity log is full. Failed to log activity for user: {user_identifier}")
        return False
    
    # Create a record with user identifier and task
    record = create_activity_record(user_identifier, task)
    
    # Append the record to the activity log
    append_to_log(activity_log, record)
    
    logger.info(f"Successfully logged activity for user: {user_identifier}")
    return True
