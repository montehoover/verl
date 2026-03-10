from queue import Queue

task_queue = Queue()

def enqueue_task(identifier: str, details: dict) -> bool:
    """
    Manages task addition to a shared queue in a concurrent processing environment.

    This function is responsible for integrating new tasks into a global queue
    for future execution.

    Args:
        identifier: A unique string that serves as an identifier for each task.
        details: A dictionary containing the details and parameters associated 
                 with the task.

    Returns:
        Indicates whether the task was successfully added to the queue (True) 
        or not (False).
    """
    try:
        task_item = (identifier, details)
        task_queue.put(task_item)
        return True
    except Exception:
        # Log the exception here if logging is set up
        return False
