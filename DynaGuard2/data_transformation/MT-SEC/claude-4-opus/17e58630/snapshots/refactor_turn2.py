from queue import Queue

task_queue = Queue()

def prepare_task(identifier: str, details: dict) -> dict:
    """
    Prepares a task object from the given identifier and details.
    
    Args:
        identifier: A unique string that serves as an identifier for each task.
        details: A dictionary containing the details and parameters associated with the task.
    
    Returns:
        dict: A task object with 'id' and 'details' fields.
    """
    return {
        'id': identifier,
        'details': details
    }

def add_to_queue(task: dict, queue: Queue) -> bool:
    """
    Adds a task to the specified queue.
    
    Args:
        task: The task object to be added to the queue.
        queue: The queue to which the task should be added.
    
    Returns:
        bool: True if the task was successfully added, False otherwise.
    """
    try:
        queue.put(task)
        return True
    except Exception:
        return False

def enqueue_task(identifier: str, details: dict) -> bool:
    """
    Manages task addition to a shared queue in a concurrent processing environment.
    
    Args:
        identifier: A unique string that serves as an identifier for each task.
        details: A dictionary containing the details and parameters associated with the task.
    
    Returns:
        bool: Indicates whether the task was successfully added to the queue (True) or not (False).
    """
    task = prepare_task(identifier, details)
    return add_to_queue(task, task_queue)
