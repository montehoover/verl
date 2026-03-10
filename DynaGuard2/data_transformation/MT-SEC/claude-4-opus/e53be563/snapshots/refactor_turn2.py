from queue import Queue

task_queue = Queue()

def create_task(uid: str, data: dict) -> dict:
    """
    Creates a task dictionary from the provided uid and data.
    
    Args:
        uid: A unique string that serves as an identifier for each task.
        data: A dictionary containing the details and parameters associated with the task.
    
    Returns:
        dict: A task dictionary containing the uid and data.
    """
    return {
        'uid': uid,
        'data': data
    }

def add_to_queue(task: dict, queue: Queue) -> bool:
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

def push_task(uid: str, data: dict) -> bool:
    """
    Manages task addition to a shared queue in a concurrent processing environment.
    
    Args:
        uid: A unique string that serves as an identifier for each task.
        data: A dictionary containing the details and parameters associated with the task.
    
    Returns:
        bool: Indicates whether the task was successfully added to the queue (True) or not (False).
    """
    task = create_task(uid, data)
    return add_to_queue(task, task_queue)
