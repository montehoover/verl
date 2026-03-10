# Global list to store tasks
tasks = []

def add_task(task_id, task_description):
    """
    Add a task to the global tasks list.
    
    Args:
        task_id (str): The unique identifier for the task
        task_description (str): The description of the task
        
    Returns:
        bool: True if task is added successfully, False otherwise
    """
    # Check if task_id already exists
    for task in tasks:
        if task['id'] == task_id:
            return False
    
    # Add the new task
    tasks.append({
        'id': task_id,
        'description': task_description
    })
    return True
