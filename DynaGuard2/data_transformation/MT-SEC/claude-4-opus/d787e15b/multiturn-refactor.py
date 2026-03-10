from queue import Queue

task_queue = Queue()


def prepare_task(task_reference: str, parameters: dict) -> dict:
    """
    Prepares a task dictionary from the given reference and parameters.
    
    This function creates a standardized task dictionary structure that can be
    used throughout the task processing system. It ensures all tasks have a
    consistent format before being added to the queue.
    
    Args:
        task_reference: A unique string that serves as an identifier for each task.
                       This reference should be unique across all tasks to enable
                       proper tracking and management.
        parameters: A dictionary containing the details and parameters associated
                   with the task. This can include any task-specific configuration
                   or data needed for execution.
    
    Returns:
        dict: A task dictionary containing the reference and parameters in a
              standardized format with keys 'reference' and 'parameters'.
    
    Example:
        >>> task = prepare_task("task_001", {"action": "process", "priority": 1})
        >>> print(task)
        {'reference': 'task_001', 'parameters': {'action': 'process', 'priority': 1}}
    """
    return {
        'reference': task_reference,
        'parameters': parameters
    }


def add_task_to_queue(task: dict, queue: Queue) -> bool:
    """
    Adds a task to the specified queue.
    
    This function attempts to add a prepared task dictionary to the given queue.
    It handles any potential exceptions that might occur during the queueing
    process and returns a boolean indicating success or failure.
    
    Args:
        task: The task dictionary to add to the queue. Should be in the format
              returned by prepare_task() with 'reference' and 'parameters' keys.
        queue: The queue to add the task to. Must be a Queue instance that
               supports the put() method.
    
    Returns:
        bool: True if the task was successfully added to the queue,
              False if any exception occurred during the queueing process.
    
    Note:
        This function catches all exceptions to ensure graceful error handling.
        In a production environment, you may want to log specific exceptions
        for debugging purposes.
    """
    try:
        queue.put(task)
        return True
    except Exception:
        return False


def register_new_task(task_reference: str, parameters: dict) -> bool:
    """
    Manages task addition to a shared queue in a concurrent processing environment.
    
    This is the main entry point for registering new tasks in the system. It
    orchestrates the task preparation and queueing process by utilizing the
    helper functions prepare_task() and add_task_to_queue(). This function
    maintains the original interface while delegating specific responsibilities
    to specialized functions for better maintainability.
    
    Args:
        task_reference: A unique string that serves as an identifier for each task.
                       This reference is crucial for tracking tasks throughout
                       their lifecycle in the processing system.
        parameters: A dictionary containing the details and parameters associated
                   with the task. The structure and content of this dictionary
                   depend on the specific task type and requirements.
    
    Returns:
        bool: Indicates whether the task was successfully added to the queue
              (True) or not (False). A False return value indicates that the
              task could not be queued and should be handled accordingly.
    
    Example:
        >>> success = register_new_task("report_123", {"type": "monthly", "user_id": 42})
        >>> if success:
        ...     print("Task registered successfully")
        ... else:
        ...     print("Failed to register task")
    """
    task = prepare_task(task_reference, parameters)
    return add_task_to_queue(task, task_queue)
