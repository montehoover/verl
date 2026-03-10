from queue import Queue

task_queue = Queue()


def prepare_task(identifier: str, details: dict) -> dict:
    """
    Prepares a task object from the given identifier and details.
    
    This function creates a standardized task dictionary that combines a unique
    identifier with its associated details, preparing it for queue insertion.
    
    Args:
        identifier: A unique string that serves as an identifier for each task.
        details: A dictionary containing the details and parameters associated 
            with the task.
    
    Returns:
        dict: A task object with 'id' and 'details' fields.
    
    Example:
        >>> task = prepare_task("task-001", {"priority": 1, "action": "process"})
        >>> print(task)
        {'id': 'task-001', 'details': {'priority': 1, 'action': 'process'}}
    """
    return {
        'id': identifier,
        'details': details
    }


def add_to_queue(task: dict, queue: Queue) -> bool:
    """
    Adds a task to the specified queue.
    
    This function safely attempts to add a task to a queue, handling any
    potential exceptions that might occur during the queue operation.
    
    Args:
        task: The task object to be added to the queue.
        queue: The queue to which the task should be added.
    
    Returns:
        bool: True if the task was successfully added, False otherwise.
    
    Example:
        >>> q = Queue()
        >>> task = {'id': 'task-001', 'details': {}}
        >>> result = add_to_queue(task, q)
        >>> print(result)
        True
    """
    try:
        queue.put(task)
        return True
    except Exception:
        return False


def enqueue_task(identifier: str, details: dict) -> bool:
    """
    Manages task addition to a shared queue in a concurrent processing environment.
    
    This function serves as the main entry point for adding tasks to the global
    task queue. It handles the complete process of task preparation and queue
    insertion by leveraging the pipeline pattern through separate functions.
    
    Args:
        identifier: A unique string that serves as an identifier for each task.
        details: A dictionary containing the details and parameters associated 
            with the task.
    
    Returns:
        bool: Indicates whether the task was successfully added to the queue 
            (True) or not (False).
    
    Example:
        >>> success = enqueue_task("task-001", {"priority": 1, "action": "process"})
        >>> if success:
        ...     print("Task added successfully")
        ... else:
        ...     print("Failed to add task")
    """
    task = prepare_task(identifier, details)
    return add_to_queue(task, task_queue)
