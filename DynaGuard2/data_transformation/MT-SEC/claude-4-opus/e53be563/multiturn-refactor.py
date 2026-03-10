from queue import Queue


task_queue = Queue()


def create_task(uid: str, data: dict) -> dict:
    """
    Creates a task dictionary from the provided uid and data.
    
    This function constructs a standardized task object that can be
    processed by the task queue system. It combines the unique identifier
    with the task data into a single dictionary structure.
    
    Args:
        uid (str): A unique string that serves as an identifier for each task.
                   This should be globally unique to prevent task collisions.
        data (dict): A dictionary containing the details and parameters 
                     associated with the task. The structure of this dictionary
                     depends on the specific task requirements.
    
    Returns:
        dict: A task dictionary containing the following keys:
              - 'uid': The unique identifier for the task
              - 'data': The task data dictionary
    
    Example:
        >>> task = create_task("task_123", {"action": "process", "priority": 1})
        >>> print(task)
        {'uid': 'task_123', 'data': {'action': 'process', 'priority': 1}}
    """
    return {
        'uid': uid,
        'data': data
    }


def add_to_queue(task: dict, queue: Queue) -> bool:
    """
    Adds a task to the specified queue.
    
    This function attempts to add a task dictionary to a queue object.
    It provides error handling to ensure that the operation doesn't
    crash the system if the queue operation fails.
    
    Args:
        task (dict): The task dictionary to add to the queue. Should contain
                     at minimum 'uid' and 'data' keys as created by create_task().
        queue (Queue): The queue to add the task to. This should be a
                       thread-safe Queue instance from the queue module.
    
    Returns:
        bool: True if the task was successfully added to the queue,
              False if any exception occurred during the queue operation.
    
    Note:
        The Queue.put() method is typically blocking and will wait if the
        queue is full (unless initialized with a maxsize). Exceptions are
        rare but could occur if the queue is closed or corrupted.
    """
    try:
        queue.put(task)
        return True
    except Exception:
        return False


def push_task(uid: str, data: dict) -> bool:
    """
    Manages task addition to a shared queue in a concurrent processing environment.
    
    This is the main entry point for adding tasks to the global task queue.
    It coordinates the creation of a properly formatted task and its addition
    to the queue, providing a simple interface for task submission.
    
    The function ensures thread-safe task addition in concurrent environments
    where multiple processes or threads may be adding tasks simultaneously.
    
    Args:
        uid (str): A unique string that serves as an identifier for each task.
                   This identifier should be unique across all tasks to enable
                   proper task tracking and management.
        data (dict): A dictionary containing the details and parameters 
                     associated with the task. The exact structure depends on
                     the task processing system's requirements.
    
    Returns:
        bool: Indicates whether the task was successfully added to the queue
              (True) or if the addition failed for any reason (False).
    
    Example:
        >>> success = push_task("user_123_task_456", {"type": "email", "to": "user@example.com"})
        >>> if success:
        ...     print("Task queued successfully")
        ... else:
        ...     print("Failed to queue task")
    """
    task = create_task(uid, data)
    return add_to_queue(task, task_queue)
