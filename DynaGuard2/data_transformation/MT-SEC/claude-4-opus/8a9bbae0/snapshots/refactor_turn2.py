from queue import Queue

task_queue = Queue()

def validate_task(task_identifier: str, task_details: dict) -> bool:
    """
    Validates the task parameters.
    
    Args:
        task_identifier: A unique string identifier assigned to each task.
        task_details: A dictionary encompassing the task's specifics and parameters.
    
    Returns:
        True if the task is valid, False otherwise.
    """
    if not isinstance(task_identifier, str) or not task_identifier:
        return False
    if not isinstance(task_details, dict):
        return False
    return True

def create_task_entry(task_identifier: str, task_details: dict) -> dict:
    """
    Creates a task entry for the queue.
    
    Args:
        task_identifier: A unique string identifier assigned to each task.
        task_details: A dictionary encompassing the task's specifics and parameters.
    
    Returns:
        A dictionary containing the task identifier and details.
    """
    return {
        'identifier': task_identifier,
        'details': task_details
    }

def add_to_queue(task_entry: dict, queue: Queue) -> bool:
    """
    Adds a task entry to the specified queue.
    
    Args:
        task_entry: The task entry to add to the queue.
        queue: The queue to add the task to.
    
    Returns:
        True if successfully added, False otherwise.
    """
    try:
        queue.put(task_entry)
        return True
    except Exception:
        return False

def enqueue_task(task_identifier: str, task_details: dict) -> bool:
    """
    Appends new tasks to a global queue for subsequent processing.
    
    Args:
        task_identifier: A unique string identifier assigned to each task.
        task_details: A dictionary encompassing the task's specifics and parameters.
    
    Returns:
        Returns True if the task is successfully enqueued, False if it fails to be added.
    """
    if not validate_task(task_identifier, task_details):
        return False
    
    task_entry = create_task_entry(task_identifier, task_details)
    return add_to_queue(task_entry, task_queue)
