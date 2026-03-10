from queue import Queue

task_queue = Queue()

def insert_task_to_queue(task_identifier: str, task_content: dict) -> bool:
    """
    Appends new tasks to a global queue for subsequent processing.
    
    Args:
        task_identifier: A unique string identifier assigned to each task.
        task_content: A dictionary encompassing the task's specifics and parameters.
    
    Returns:
        bool: Returns True if the task is successfully enqueued, False if it fails to be added.
    """
    try:
        # Create a task object that includes both the identifier and content
        task = {
            'id': task_identifier,
            'content': task_content
        }
        
        # Add the task to the queue
        task_queue.put(task)
        
        return True
    except Exception:
        # If any error occurs during enqueuing, return False
        return False
