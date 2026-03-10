from queue import Queue

task_queue = Queue()

def register_new_task(task_reference: str, parameters: dict) -> bool:
    """
    Manages task addition to a shared queue in a concurrent processing environment.
    
    Args:
        task_reference: A unique string that serves as an identifier for each task.
        parameters: A dictionary containing the details and parameters associated with the task.
    
    Returns:
        bool: Indicates whether the task was successfully added to the queue (True) or not (False).
    """
    try:
        task = {
            'reference': task_reference,
            'parameters': parameters
        }
        task_queue.put(task)
        return True
    except Exception:
        return False
