# Global list to store tasks
tasks = []

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
