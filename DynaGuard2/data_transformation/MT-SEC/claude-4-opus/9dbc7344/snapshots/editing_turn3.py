from queue import Queue

# Global queue for task storage and processing
task_queue = Queue()

def insert_task_to_queue(task_identifier, task_content):
    """
    Insert a task into the global task queue.
    
    Args:
        task_identifier (str): The unique identifier for the task
        task_content (dict): Dictionary containing task parameters
    
    Returns:
        bool: True if task was inserted successfully, False otherwise
    """
    try:
        # Create task with identifier and content
        task = {
            'id': task_identifier,
            'content': task_content
        }
        
        # Put task in queue
        task_queue.put(task)
        return True
    except:
        return False
