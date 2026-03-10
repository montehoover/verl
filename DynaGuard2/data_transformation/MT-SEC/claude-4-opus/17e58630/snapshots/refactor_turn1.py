from queue import Queue

task_queue = Queue()

def enqueue_task(identifier: str, details: dict) -> bool:
    """
    Manages task addition to a shared queue in a concurrent processing environment.
    
    Args:
        identifier: A unique string that serves as an identifier for each task.
        details: A dictionary containing the details and parameters associated with the task.
    
    Returns:
        bool: Indicates whether the task was successfully added to the queue (True) or not (False).
    """
    try:
        task = {
            'id': identifier,
            'details': details
        }
        task_queue.put(task)
        return True
    except Exception:
        return False
