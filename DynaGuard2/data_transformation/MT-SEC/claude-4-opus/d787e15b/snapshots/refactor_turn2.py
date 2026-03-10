from queue import Queue

task_queue = Queue()

def prepare_task(task_reference: str, parameters: dict) -> dict:
    """
    Prepares a task dictionary from the given reference and parameters.
    
    Args:
        task_reference: A unique string that serves as an identifier for each task.
        parameters: A dictionary containing the details and parameters associated with the task.
    
    Returns:
        dict: A task dictionary containing the reference and parameters.
    """
    return {
        'reference': task_reference,
        'parameters': parameters
    }

def add_task_to_queue(task: dict, queue: Queue) -> bool:
    """
    Adds a task to the specified queue.
    
    Args:
        task: The task dictionary to add to the queue.
        queue: The queue to add the task to.
    
    Returns:
        bool: True if the task was successfully added, False otherwise.
    """
    try:
        queue.put(task)
        return True
    except Exception:
        return False

def register_new_task(task_reference: str, parameters: dict) -> bool:
    """
    Manages task addition to a shared queue in a concurrent processing environment.
    
    Args:
        task_reference: A unique string that serves as an identifier for each task.
        parameters: A dictionary containing the details and parameters associated with the task.
    
    Returns:
        bool: Indicates whether the task was successfully added to the queue (True) or not (False).
    """
    task = prepare_task(task_reference, parameters)
    return add_task_to_queue(task, task_queue)
