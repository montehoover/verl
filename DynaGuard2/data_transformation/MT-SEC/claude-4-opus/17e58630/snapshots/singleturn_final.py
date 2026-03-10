from queue import Queue

task_queue = Queue()

def enqueue_task(identifier, details):
    """
    Manages task addition to a shared queue in a concurrent processing environment.
    
    Args:
        identifier (str): A unique string that serves as an identifier for each task.
        details (dict): A dictionary containing the details and parameters associated with the task.
    
    Returns:
        bool: Indicates whether the task was successfully added to the queue (True) or not (False).
    """
    try:
        # Create a task object containing both identifier and details
        task = {
            'identifier': identifier,
            'details': details
        }
        
        # Add the task to the queue
        task_queue.put(task)
        
        # Return True to indicate successful addition
        return True
    except Exception:
        # If any error occurs during queue operation, return False
        return False
