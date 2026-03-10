from queue import Queue

# Global queue to store tasks
task_queue = Queue()

def push_task(uid, data):
    """
    Add a task to the task queue for concurrent processing.
    
    Args:
        uid (str): The unique identifier for the task
        data (dict): Task parameters and information
        
    Returns:
        bool: True if task is added successfully, False otherwise
    """
    try:
        task_queue.put({
            'uid': uid,
            'data': data
        })
        return True
    except Exception:
        return False
