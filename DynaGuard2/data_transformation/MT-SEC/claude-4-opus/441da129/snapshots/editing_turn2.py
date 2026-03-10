import datetime
import os

def log_system_event(event_type, event_description, timestamp):
    """
    Logs system events to a file with timestamp.
    
    Args:
        event_type (str): The type of event being logged
        event_description (str): Description of the event
        timestamp (datetime.datetime): When the event occurred
        
    Returns:
        bool: True if logged successfully, False otherwise
    """
    try:
        # Create logs directory if it doesn't exist
        log_dir = "logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # Generate log filename with current date
        log_filename = os.path.join(log_dir, f"system_events_{datetime.date.today()}.log")
        
        # Format the log entry using the provided timestamp
        timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp_str}] {event_type}: {event_description}\n"
        
        # Append to log file
        with open(log_filename, 'a', encoding='utf-8') as log_file:
            log_file.write(log_entry)
        
        return True
        
    except Exception as e:
        # If any error occurs during logging, return False
        return False
