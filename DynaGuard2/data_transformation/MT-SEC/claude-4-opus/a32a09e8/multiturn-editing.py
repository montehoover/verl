from queue import Queue

# Global list to store tasks
tasks = []

# Global queue for parallel processing
task_queue = Queue()

def add_task(task_name, task_info, task_details):
    """Add a task to the global tasks list.
    
    Args:
        task_name (str): The name of the task
        task_info (str): Additional information about the task
        task_details (dict): Dictionary containing priority, deadline, and other details
        
    Returns:
        bool: True if task added successfully, False otherwise
    """
    try:
        task = {
            'name': task_name,
            'info': task_info,
            'details': task_details
        }
        tasks.append(task)
        return True
    except:
        return False

def insert_task_to_queue(job_id, job_data):
    """Insert a task into the global task queue for parallel processing.
    
    Args:
        job_id (str): Unique identifier for the task
        job_data (dict): Dictionary containing task details
        
    Returns:
        bool: True if task added successfully, False otherwise
    """
    try:
        task = {
            'id': job_id,
            'data': job_data
        }
        task_queue.put(task)
        return True
    except:
        return False
