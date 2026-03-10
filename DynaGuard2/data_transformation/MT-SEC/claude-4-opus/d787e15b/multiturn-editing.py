from queue import Queue

# Global list to store tasks
tasks = []

# Global queue for concurrent task management
task_queue = Queue()

def add_task(task_id, task_description, task_info):
    """
    Add a task to the global tasks list.
    
    Args:
        task_id (str): The unique identifier for the task
        task_description (str): The description of the task
        task_info (dict): Additional task information (e.g., priority, deadline)
        
    Returns:
        bool: True if task added successfully, False otherwise
    """
    # Check if task_id already exists
    for task in tasks:
        if task['id'] == task_id:
            return False
    
    # Add the new task
    tasks.append({
        'id': task_id,
        'description': task_description,
        'info': task_info
    })
    return True

def register_new_task(task_reference, parameters):
    """
    Register a new task in the concurrent task queue.
    
    Args:
        task_reference (str): The unique task ID
        parameters (dict): Task details
        
    Returns:
        bool: True if task registered successfully, False otherwise
    """
    try:
        # Create task object with reference and parameters
        task = {
            'reference': task_reference,
            'parameters': parameters
        }
        
        # Add task to the queue
        task_queue.put(task)
        return True
    except:
        return False
