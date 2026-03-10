from queue import Queue

task_queue = Queue()

def create_task(task_identifier: str, task_content: dict) -> dict:
    """
    Creates a task dictionary from the given identifier and content.
    
    Args:
        task_identifier: A unique string identifier assigned to each task.
        task_content: A dictionary encompassing the task's specifics and parameters.
    
    Returns:
        dict: A task dictionary with 'id' and 'content' keys.
    """
    return {
        'id': task_identifier,
        'content': task_content
    }

def enqueue_task(task: dict, queue: Queue) -> bool:
    """
    Adds a task to the specified queue.
    
    Args:
        task: The task dictionary to be enqueued.
        queue: The queue to add the task to.
    
    Returns:
        bool: True if the task is successfully enqueued, False otherwise.
    """
    try:
        queue.put(task)
        return True
    except Exception:
        return False

def insert_task_to_queue(task_identifier: str, task_content: dict) -> bool:
    """
    Appends new tasks to a global queue for subsequent processing.
    
    Args:
        task_identifier: A unique string identifier assigned to each task.
        task_content: A dictionary encompassing the task's specifics and parameters.
    
    Returns:
        bool: Returns True if the task is successfully enqueued, False if it fails to be added.
    """
    task = create_task(task_identifier, task_content)
    return enqueue_task(task, task_queue)
