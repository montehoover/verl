from queue import Queue

# Global queue to store tasks
task_queue = Queue()

def enqueue_task(identifier, details):
    """
    Add a task to the global task queue for concurrent processing.
    
    Args:
        identifier (str): Unique task ID
        details (dict): Dictionary containing task parameters
        
    Returns:
        bool: True if task is enqueued successfully, False otherwise
    """
    if isinstance(identifier, str) and identifier.strip() and isinstance(details, dict):
        task = {
            'id': identifier.strip(),
            'details': details
        }
        try:
            task_queue.put(task)
            return True
        except:
            return False
    return False
