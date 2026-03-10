from queue import Queue

# Global queue to store tasks
task_queue = Queue()

def enqueue_task(task_identifier, task_details):
    """
    Add a task to the global task queue.
    
    Args:
        task_identifier (str): The unique identifier for the task
        task_details (dict): Dictionary containing task specifics
        
    Returns:
        bool: True if task is added successfully, False otherwise
    """
    try:
        # Create task object with identifier and details
        task = {
            'id': task_identifier,
            **task_details
        }
        
        # Add task to queue
        task_queue.put(task)
        return True
    except Exception:
        return False
