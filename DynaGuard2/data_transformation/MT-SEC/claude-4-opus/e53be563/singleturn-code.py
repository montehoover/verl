from queue import Queue

task_queue = Queue()

def push_task(uid: str, data: dict) -> bool:
    """
    Manages task addition to a shared queue in a concurrent processing environment.
    
    Args:
        uid: A unique string that serves as an identifier for each task.
        data: A dictionary containing the details and parameters associated with the task.
    
    Returns:
        bool: Indicates whether the task was successfully added to the queue (True) or not (False).
    """
    try:
        # Create a task object containing both the uid and data
        task = {
            'uid': uid,
            'data': data
        }
        
        # Add the task to the queue
        task_queue.put(task)
        
        # Return True to indicate successful addition
        return True
    except Exception:
        # If any error occurs during the process, return False
        return False
