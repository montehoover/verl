from queue import Queue

task_queue = Queue()

def enqueue_task(task_identifier: str, task_details: dict) -> bool:
    """
    Appends new tasks to a global queue for subsequent processing.
    
    Args:
        task_identifier: A unique string identifier assigned to each task.
        task_details: A dictionary encompassing the task's specifics and parameters.
    
    Returns:
        bool: Returns True if the task is successfully enqueued, False if it fails to be added.
    """
    try:
        # Create a task object containing both the identifier and details
        task = {
            'id': task_identifier,
            'details': task_details
        }
        
        # Add the task to the queue
        task_queue.put(task)
        
        return True
    except Exception:
        # If any exception occurs during enqueuing, return False
        return False
