from queue import Queue

task_queue = Queue()

def enqueue_task(task_identifier: str, task_details: dict) -> bool:
    """
    Appends new tasks to a global queue for subsequent processing.
    
    Args:
        task_identifier: A unique string identifier assigned to each task.
        task_details: A dictionary encompassing the task's specifics and parameters.
    
    Returns:
        Returns True if the task is successfully enqueued, False if it fails to be added.
    """
    try:
        task_queue.put({
            'identifier': task_identifier,
            'details': task_details
        })
        return True
    except Exception:
        return False
